import math


class CentroidTracker:
    """Gives each detected person a persistent ID by matching
    new detections to the nearest person from the previous frame."""

    def __init__(self, max_disappeared=90, max_distance=100):
        self.next_id = 0
        self.objects = {}       # id -> (center_x, center_y)
        self.boxes = {}         # id -> (x, y, w, h)
        self.disappeared = {}   # id -> number of frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.events = []        # filled every frame: ("entered", id) or ("left", id)

    def _register(self, centroid, box):
        self.objects[self.next_id] = centroid
        self.boxes[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.events.append(("entered", self.next_id))
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.boxes[obj_id]
        del self.disappeared[obj_id]
        self.events.append(("left", obj_id))

    def _mark_missing(self, obj_id):
        self.disappeared[obj_id] += 1
        if self.disappeared[obj_id] > self.max_disappeared:
            self._deregister(obj_id)

    def update(self, boxes):
        """boxes: list of (x, y, w, h). Returns dict of id -> box."""
        self.events = []

        if len(boxes) == 0:
            for obj_id in list(self.objects.keys()):
                self._mark_missing(obj_id)
            return self.boxes

        centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]

        if len(self.objects) == 0:
            for c, b in zip(centroids, boxes):
                self._register(c, b)
            return self.boxes

        pairs = []
        for obj_id, old_c in self.objects.items():
            for j, new_c in enumerate(centroids):
                d = math.hypot(old_c[0] - new_c[0], old_c[1] - new_c[1])
                pairs.append((d, obj_id, j))
        pairs.sort()

        used_ids, used_dets = set(), set()
        for d, obj_id, j in pairs:
            if obj_id in used_ids or j in used_dets:
                continue
            if d > self.max_distance:
                break
            self.objects[obj_id] = centroids[j]
            self.boxes[obj_id] = boxes[j]
            self.disappeared[obj_id] = 0
            used_ids.add(obj_id)
            used_dets.add(j)

        for obj_id in list(self.objects.keys()):
            if obj_id not in used_ids:
                self._mark_missing(obj_id)

        for j, (c, b) in enumerate(zip(centroids, boxes)):
            if j not in used_dets:
                self._register(c, b)

        return self.boxes