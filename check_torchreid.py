import torchreid
print(dir(torchreid.utils))
try:
    from torchreid.utils import FeatureExtractor
    print("FeatureExtractor found")
except ImportError:
    print("FeatureExtractor NOT found")
