import gdown
import zipfile
import subprocess
import urllib.request
import os

# Make folder to save data in. 
os.makedirs('./data', exist_ok=True)
    

print('Downloading SDF datasets')
gdown.download("https://drive.google.com/uc?id=1xBo6OCGmyWi0qD74EZW4lc45Gs4HXjWw", './data/gt_armadillo.xyz')
gdown.download("https://drive.google.com/uc?id=1Pm3WHUvJiMJEKUnnhMjB6mUAnR9qhnxm", './data/gt_dragon.xyz')
gdown.download("https://drive.google.com/uc?id=1wE24AZtXS8jbIIc-amYeEUtlxN8dFYCo", './data/gt_lucy.xyz')
gdown.download("https://drive.google.com/uc?id=1OVw0JNA-NZtDXVmkf57erqwqDjqmF5Mc", './data/gt_thai.xyz')


print('Done!')
