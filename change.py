import glob
import os

# l = glob.glob('/data/ugui0/YouTube8M/segment1/validate/*tfrecord')
for i in range(769, 3844):
    os.rename('/data/ugui0/YouTube8M/segment1/validate/validate{}.tfrecord'.format(str(i)),
              '/data/ugui0/YouTube8M/segment1/validate/train{}.tfrecord'.format(str(i - 769)))