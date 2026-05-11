#include <Arduino.h>
#include <FastAccelStepper.h>
#include <TMCStepper.h>

namespace {

// ------------------------------ Hardware pins ------------------------------
constexpr uint8_t TILT_DIR_PIN = 26;
constexpr uint8_t TILT_STEP_PIN = 27;
constexpr uint8_t PAN_DIR_PIN = 14;
constexpr uint8_t PAN_STEP_PIN = 12;
constexpr uint8_t ENABLE_PIN = 13;  // TMC2209 EN/ENN is active low.

constexpr uint8_t TMC_UART_RX_PIN = 16;
constexpr uint8_t TMC_UART_TX_PIN = 17;

// ------------------------------ Driver config ------------------------------
constexpr uint8_t PAN_TMC_ADDRESS = 0;
constexpr uint8_t TILT_TMC_ADDRESS = 1;
constexpr float R_SENSE_OHMS = 0.11f;

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t TMC_UART_BAUD = 115200;
constexpr uint16_t DEFAULT_MICROSTEPS = 16;
constexpr uint16_t RMS_CURRENT_MA = 800;

// ------------------------------ Motion policy ------------------------------
constexpr int32_t DEFAULT_MAX_SPEED_STEPS_PER_SEC = 20000;
constexpr int32_t MIN_MAX_SPEED_STEPS_PER_SEC = 100;
constexpr int32_t HARD_MAX_SPEED_STEPS_PER_SEC = 80000;
constexpr int32_t ACCELERATION_STEPS_PER_SEC2 = 50000;
constexpr uint32_t COMMAND_TIMEOUT_MS = 500;

// Flip either value if positive Jetson velocity is mechanically backwards.
constexpr bool PAN_DIR_HIGH_COUNTS_UP = true;
constexpr bool TILT_DIR_HIGH_COUNTS_UP = true;

constexpr uint32_t MOTION_LOOP_PERIOD_MS = 2;
constexpr uint32_t SERIAL_TASK_PERIOD_MS = 1;
constexpr uint16_t DRIVER_ENABLE_DELAY_US = 1000;
constexpr uint16_t DRIVER_DISABLE_DELAY_MS = 250;
constexpr uint16_t DIR_CHANGE_DELAY_US = 200;
constexpr size_t SERIAL_LINE_BUFFER_SIZE = 64;

struct VelocityCommand {
  int32_t panStepsPerSec;
  int32_t tiltStepsPerSec;
};

FastAccelStepperEngine stepperEngine;
FastAccelStepper *panStepper = nullptr;
FastAccelStepper *tiltStepper = nullptr;

TMC2209Stepper panDriver(&Serial1, R_SENSE_OHMS, PAN_TMC_ADDRESS);
TMC2209Stepper tiltDriver(&Serial1, R_SENSE_OHMS, TILT_TMC_ADDRESS);

portMUX_TYPE commandMux = portMUX_INITIALIZER_UNLOCKED;
volatile int32_t targetPanStepsPerSec = 0;
volatile int32_t targetTiltStepsPerSec = 0;
volatile uint32_t lastValidCommandMs = 0;
volatile int32_t configuredMaxSpeedStepsPerSec = DEFAULT_MAX_SPEED_STEPS_PER_SEC;

int32_t appliedPanStepsPerSec = INT32_MIN;
int32_t appliedTiltStepsPerSec = INT32_MIN;
uint16_t configuredMicrosteps = DEFAULT_MICROSTEPS;

uint16_t normalizeDriverMicrosteps(const uint16_t microsteps) {
  // TMCStepper 0.7.3 uses microsteps(0) to select full-step mode.
  return microsteps == 1 ? 0 : microsteps;
}

uint16_t displayDriverMicrosteps(const uint16_t driverMicrosteps) {
  return driverMicrosteps == 0 ? 1 : driverMicrosteps;
}

int32_t clampSpeed(const int32_t requestedStepsPerSec) {
  int32_t maxSpeed = DEFAULT_MAX_SPEED_STEPS_PER_SEC;
  portENTER_CRITICAL(&commandMux);
  maxSpeed = configuredMaxSpeedStepsPerSec;
  portEXIT_CRITICAL(&commandMux);

  if (requestedStepsPerSec > maxSpeed) {
    return maxSpeed;
  }
  if (requestedStepsPerSec < -maxSpeed) {
    return -maxSpeed;
  }
  return requestedStepsPerSec;
}

int32_t clampToLimit(const int32_t value, const int32_t limit) {
  if (value > limit) {
    return limit;
  }
  if (value < -limit) {
    return -limit;
  }
  return value;
}

bool parseInt32Strict(const char *begin, const char *end, int32_t *value) {
  if (begin == nullptr || end == nullptr || value == nullptr || begin >= end) {
    return false;
  }

  char *parseEnd = nullptr;
  const long parsed = strtol(begin, &parseEnd, 10);
  if (parseEnd != end) {
    return false;
  }
  if (parsed < INT32_MIN || parsed > INT32_MAX) {
    return false;
  }

  *value = static_cast<int32_t>(parsed);
  return true;
}

bool isValidMicrosteps(const int32_t microsteps) {
  switch (microsteps) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
    case 256:
      return true;
    default:
      return false;
  }
}

void forceStopTargets() {
  portENTER_CRITICAL(&commandMux);
  targetPanStepsPerSec = 0;
  targetTiltStepsPerSec = 0;
  lastValidCommandMs = millis();
  portEXIT_CRITICAL(&commandMux);
}

void setMaxSpeedLimit(const int32_t requestedMaxSpeed) {
  const int32_t maxSpeed =
      constrain(requestedMaxSpeed, MIN_MAX_SPEED_STEPS_PER_SEC,
                HARD_MAX_SPEED_STEPS_PER_SEC);

  portENTER_CRITICAL(&commandMux);
  configuredMaxSpeedStepsPerSec = maxSpeed;
  targetPanStepsPerSec = clampToLimit(targetPanStepsPerSec, maxSpeed);
  targetTiltStepsPerSec = clampToLimit(targetTiltStepsPerSec, maxSpeed);
  portEXIT_CRITICAL(&commandMux);

  Serial.printf("OK S:%ld\r\n", static_cast<long>(maxSpeed));
}

void setMicrosteps(const uint16_t microsteps) {
  if (microsteps == configuredMicrosteps) {
    Serial.printf("OK M:%u\r\n", microsteps);
    return;
  }

  // Stop before changing microstep resolution so one host velocity unit remains
  // unambiguous after the driver register changes.
  forceStopTargets();
  if (panStepper != nullptr) {
    panStepper->stopMove();
  }
  if (tiltStepper != nullptr) {
    tiltStepper->stopMove();
  }

  const uint16_t driverMicrosteps = normalizeDriverMicrosteps(microsteps);
  panDriver.mstep_reg_select(true);
  tiltDriver.mstep_reg_select(true);
  panDriver.microsteps(driverMicrosteps);
  tiltDriver.microsteps(driverMicrosteps);
  configuredMicrosteps = microsteps;
  Serial.printf("OK M:%u PM:%u TM:%u\r\n", microsteps,
                displayDriverMicrosteps(panDriver.microsteps()),
                displayDriverMicrosteps(tiltDriver.microsteps()));
}

bool parseSingleIntCommand(const char *payload, int32_t *value) {
  if (payload == nullptr || value == nullptr) {
    return false;
  }
  return parseInt32Strict(payload, payload + strlen(payload), value);
}

bool parseVelocityLine(char *line, VelocityCommand *command) {
  if (line == nullptr || command == nullptr) {
    return false;
  }

  if (line[0] != 'V' || line[1] != ':') {
    return false;
  }

  char *const panBegin = line + 2;
  char *const comma = strchr(panBegin, ',');
  if (comma == nullptr) {
    return false;
  }

  char *const tiltBegin = comma + 1;
  char *const lineEnd = line + strlen(line);
  int32_t pan = 0;
  int32_t tilt = 0;

  if (!parseInt32Strict(panBegin, comma, &pan) ||
      !parseInt32Strict(tiltBegin, lineEnd, &tilt)) {
    return false;
  }

  command->panStepsPerSec = clampSpeed(pan);
  command->tiltStepsPerSec = clampSpeed(tilt);
  return true;
}

bool handleConfigLine(char *line) {
  if (line == nullptr || line[1] != ':') {
    return false;
  }

  int32_t value = 0;
  if (!parseSingleIntCommand(line + 2, &value)) {
    return false;
  }

  if (line[0] == 'M') {
    if (!isValidMicrosteps(value)) {
      Serial.printf("ERR M:%ld\r\n", static_cast<long>(value));
      return false;
    }
    setMicrosteps(static_cast<uint16_t>(value));
    return true;
  }

  if (line[0] == 'S') {
    setMaxSpeedLimit(value);
    return true;
  }

  return false;
}

void printStatus() {
  int32_t maxSpeed = DEFAULT_MAX_SPEED_STEPS_PER_SEC;
  int32_t panTarget = 0;
  int32_t tiltTarget = 0;

  portENTER_CRITICAL(&commandMux);
  maxSpeed = configuredMaxSpeedStepsPerSec;
  panTarget = targetPanStepsPerSec;
  tiltTarget = targetTiltStepsPerSec;
  portEXIT_CRITICAL(&commandMux);

  Serial.printf(
      "STAT M:%u PM:%u TM:%u S:%ld PV:0x%02X TV:0x%02X V:%ld,%ld\r\n",
      configuredMicrosteps, displayDriverMicrosteps(panDriver.microsteps()),
      displayDriverMicrosteps(tiltDriver.microsteps()),
      static_cast<long>(maxSpeed), panDriver.version(), tiltDriver.version(),
      static_cast<long>(panTarget), static_cast<long>(tiltTarget));
}

bool handleStatusLine(const char *line) {
  if (line == nullptr || strcmp(line, "?") != 0) {
    return false;
  }

  printStatus();
  return true;
}

void publishCommand(const VelocityCommand &command) {
  portENTER_CRITICAL(&commandMux);
  targetPanStepsPerSec = command.panStepsPerSec;
  targetTiltStepsPerSec = command.tiltStepsPerSec;
  lastValidCommandMs = millis();
  portEXIT_CRITICAL(&commandMux);
}

VelocityCommand snapshotCommand() {
  VelocityCommand command{};
  const uint32_t now = millis();

  portENTER_CRITICAL(&commandMux);
  if ((now - lastValidCommandMs) > COMMAND_TIMEOUT_MS) {
    targetPanStepsPerSec = 0;
    targetTiltStepsPerSec = 0;
  }

  command.panStepsPerSec = targetPanStepsPerSec;
  command.tiltStepsPerSec = targetTiltStepsPerSec;
  portEXIT_CRITICAL(&commandMux);

  return command;
}

void configureTmc2209(TMC2209Stepper &driver, const char *axisName) {
  driver.begin();

  // UART mode selects microsteps/current in registers instead of CFG pins.
  driver.pdn_disable(true);
  driver.I_scale_analog(false);
  driver.internal_Rsense(false);
  driver.mstep_reg_select(true);

  driver.toff(4);
  driver.blank_time(24);
  driver.rms_current(RMS_CURRENT_MA);
  driver.microsteps(normalizeDriverMicrosteps(configuredMicrosteps));

  driver.en_spreadCycle(false);  // false enables StealthChop on TMC2209.
  driver.pwm_autoscale(true);
  driver.pwm_autograd(true);
  driver.TPWMTHRS(0);
  driver.TPOWERDOWN(20);
  driver.semin(0);  // Disable CoolStep until the mechanical limits are known.

  const uint8_t version = driver.version();
  Serial.printf("%s TMC2209 UART version: 0x%02X\r\n", axisName, version);
}

bool configureStepper(FastAccelStepper *stepper, const char *axisName,
                      const uint8_t dirPin,
                      const bool dirHighCountsUp) {
  if (stepper == nullptr) {
    Serial.printf("%s FastAccelStepper allocation failed\r\n", axisName);
    return false;
  }

  stepper->setDirectionPin(dirPin, dirHighCountsUp, DIR_CHANGE_DELAY_US);
  stepper->setEnablePin(ENABLE_PIN, true);
  stepper->setAutoEnable(true);
  stepper->setDelayToEnable(DRIVER_ENABLE_DELAY_US);
  stepper->setDelayToDisable(DRIVER_DISABLE_DELAY_MS);
  stepper->setSpeedInHz(1);
  stepper->setAcceleration(ACCELERATION_STEPS_PER_SEC2);
  stepper->stopMove();
  return true;
}

void applyAxisSpeed(FastAccelStepper *stepper, int32_t *lastApplied,
                    const int32_t requestedStepsPerSec) {
  if (stepper == nullptr || lastApplied == nullptr ||
      requestedStepsPerSec == *lastApplied) {
    return;
  }

  if (requestedStepsPerSec == 0) {
    stepper->stopMove();
    *lastApplied = requestedStepsPerSec;
    return;
  }

  const uint32_t speedMagnitude =
      static_cast<uint32_t>(abs(requestedStepsPerSec));

  if (stepper->setSpeedInHz(speedMagnitude) != 0) {
    Serial.printf("Rejected speed command: %ld step/s\r\n",
                  static_cast<long>(requestedStepsPerSec));
    return;
  }

  stepper->applySpeedAcceleration();
  if (requestedStepsPerSec > 0) {
    stepper->runForward();
  } else {
    stepper->runBackward();
  }

  *lastApplied = requestedStepsPerSec;
}

void applyVelocityTargets(const VelocityCommand &command) {
  applyAxisSpeed(panStepper, &appliedPanStepsPerSec,
                 command.panStepsPerSec);
  applyAxisSpeed(tiltStepper, &appliedTiltStepsPerSec,
                 command.tiltStepsPerSec);
}

void serialCommandTask(void *parameter) {
  (void)parameter;

  char line[SERIAL_LINE_BUFFER_SIZE] = {};
  size_t index = 0;
  bool discardingOverflow = false;

  for (;;) {
    while (Serial.available() > 0) {
      const char incoming = static_cast<char>(Serial.read());

      if (incoming == '\r') {
        continue;
      }

      if (incoming == '\n') {
        if (!discardingOverflow && index > 0) {
          line[index] = '\0';

          VelocityCommand command{};
          if (parseVelocityLine(line, &command)) {
            publishCommand(command);
          } else if (handleStatusLine(line)) {
            // Status requests are intentionally not fail-safe heartbeats.
          } else {
            handleConfigLine(line);
          }
        }

        index = 0;
        discardingOverflow = false;
        continue;
      }

      if (discardingOverflow) {
        continue;
      }

      if (index < (SERIAL_LINE_BUFFER_SIZE - 1)) {
        line[index++] = incoming;
      } else {
        index = 0;
        discardingOverflow = true;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(SERIAL_TASK_PERIOD_MS));
  }
}

}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial1.begin(TMC_UART_BAUD, SERIAL_8N1, TMC_UART_RX_PIN, TMC_UART_TX_PIN);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, HIGH);

  Serial.println("AIMBOT ESP32 pan/tilt motor controller booting");

  configureTmc2209(panDriver, "Pan");
  configureTmc2209(tiltDriver, "Tilt");

  stepperEngine.init();

  panStepper = stepperEngine.stepperConnectToPin(PAN_STEP_PIN);
  tiltStepper = stepperEngine.stepperConnectToPin(TILT_STEP_PIN);

  const bool panReady = configureStepper(
      panStepper, "Pan", PAN_DIR_PIN, PAN_DIR_HIGH_COUNTS_UP);
  const bool tiltReady = configureStepper(
      tiltStepper, "Tilt", TILT_DIR_PIN, TILT_DIR_HIGH_COUNTS_UP);

  portENTER_CRITICAL(&commandMux);
  lastValidCommandMs = millis();
  portEXIT_CRITICAL(&commandMux);

  xTaskCreatePinnedToCore(serialCommandTask, "SerialCommandTask", 4096, nullptr,
                          2, nullptr, 0);

  Serial.printf("Stepper init: pan=%s tilt=%s\r\n", panReady ? "ok" : "fail",
                tiltReady ? "ok" : "fail");
}

void loop() {
  applyVelocityTargets(snapshotCommand());
  vTaskDelay(pdMS_TO_TICKS(MOTION_LOOP_PERIOD_MS));
}
