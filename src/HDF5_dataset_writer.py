import h5py
import os


class HDF5DatasetWriter():
    def __init__(self, image_dims, depth_map_dims, outputPath, bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.images = self.db.create_dataset(name="images", shape=image_dims, dtype="float")
        self.depth_maps = self.db.create_dataset(name="depth_maps", shape=depth_map_dims, dtype='float')
        self.bufSize = bufSize
        self.buffer = {"images": [], "depth_maps": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["images"].extend(rows)
        self.buffer["depth_maps"].extend(labels)
        if len(self.buffer["images"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["images"])
        self.images[self.idx:i] = self.buffer["images"]
        self.depth_maps[self.idx:i] = self.buffer["depth_maps"]
        self.idx = i
        self.buffer = {"images": [], "depth_maps": []}

    def close(self):
        if len(self.buffer["images"]) > 0:
            self.flush()
        self.db.close()