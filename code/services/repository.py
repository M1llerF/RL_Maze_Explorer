import os
import pickle
import hashlib
import shutil
import tempfile
import threading
import time
from typing import Any, cast


ProfileDict = dict[str, Any]
HeatmapData = dict[tuple[int, int], int]


_STATS_KEYS: frozenset[str] = frozenset({
    'total_steps', 'non_repeating_steps_taken',
    'times_revisited_squares', 'times_hit_wall', 'times_hit_enemy',
})


class ArtifactsRepository:
    """Centralized persistence for training artifacts under profiles/<name>.

    Scope:
    - Q-table read/write (+ checksum)
    - Rewards log append
    - Maze runs (mazes.json) latest/highest/lowest
    - Profile aggregated stats (stats.pkl) counters

    profile.pkl is owned exclusively by ProfileManager. This class never
    writes to it. Stats use stats.pkl to prevent collisions.

    All writes are atomic.
    """

    def __init__(self, baseDir: str = "profiles"):
        self.baseDir = baseDir

    _pathLocks: dict[str, threading.Lock] = {}
    _pathLocksGuard = threading.Lock()

    # ---------- Helpers ----------
    def _profileDir(self, profile: str) -> str:
        return os.path.join(self.baseDir, profile)

    def _ensureDir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @classmethod
    def _lockForPath(cls, path: str) -> threading.Lock:
        normalized = os.path.abspath(path)
        with cls._pathLocksGuard:
            lock = cls._pathLocks.get(normalized)
            if lock is None:
                lock = threading.Lock()
                cls._pathLocks[normalized] = lock
            return lock

    def _atomicReplace(self, tempPath: str, path: str) -> None:
        lock = self._lockForPath(path)
        with lock:
            last_error: OSError | None = None
            for delay in (0.0, 0.01, 0.05, 0.1, 0.25):
                if delay > 0.0:
                    time.sleep(delay)
                try:
                    os.replace(tempPath, path)
                    return
                except PermissionError as exc:
                    last_error = exc
            raise cast(OSError, last_error)

    # ---------- Generic paths ----------
    def _profilePath(self, profile: str) -> str:
        """Profile config path. Owned exclusively by ProfileManager — never write stats here."""
        return os.path.join(self._profileDir(profile), "profile.pkl")

    def _statsPath(self, profile: str) -> str:
        """Runtime training statistics. Separate from profile.pkl to prevent config collisions."""
        return os.path.join(self._profileDir(profile), "stats.pkl")

    def rewardsPath(self, profile: str) -> str:
        return os.path.join(self._profileDir(profile), "SimulationRewards.txt")

    def mazesJsonPath(self, profile: str) -> str:
        return os.path.join(self._profileDir(profile), "mazes.json")

    # ---------- Q-table ----------
    def qTablePath(self, profile: str) -> str:
        return os.path.join(self._profileDir(profile), "q_table.pkl")

    def qTableChecksumPath(self, profile: str) -> str:
        return os.path.join(self._profileDir(profile), "q_table.checksum")

    # ---------- Future model artifacts (extension seam) ----------
    def modelArtifactsDir(self, profile: str) -> str:
        return os.path.join(self._profileDir(profile), "artifacts")

    def _validateArtifactName(self, artifactName: str) -> str:
        # Fail fast on path traversal or empty names.
        cleaned = artifactName.strip()
        if not cleaned or os.path.basename(cleaned) != cleaned:
            raise ValueError(f"Invalid artifact name: {artifactName}")
        return cleaned

    def modelArtifactPath(self, profile: str, artifactName: str) -> str:
        name = self._validateArtifactName(artifactName)
        return os.path.join(self.modelArtifactsDir(profile), name)

    def saveModelArtifactBytes(self, profile: str, artifactName: str, data: bytes) -> None:
        d = self.modelArtifactsDir(profile)
        self._ensureDir(d)
        path = self.modelArtifactPath(profile, artifactName)
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='wb') as tmp:
            tmp.write(data)
            temp = tmp.name
        self._atomicReplace(temp, path)

    def loadModelArtifactBytes(self, profile: str, artifactName: str) -> bytes | None:
        path = self.modelArtifactPath(profile, artifactName)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        try:
            with open(path, 'rb') as f:
                return f.read()
        except OSError:
            return None

    def _fileChecksum(self, filePath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filePath, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def saveQTable(self, profile: str, qTable: dict[Any, Any]) -> None:
        d = self._profileDir(profile)
        self._ensureDir(d)
        path = self.qTablePath(profile)
        # Atomic write
        with tempfile.NamedTemporaryFile(delete=False, dir=d) as tmp:
            pickle.dump(qTable, tmp)
            tempName = tmp.name
        self._atomicReplace(tempName, path)
        # Write checksum
        chksum = self._fileChecksum(path)
        with open(self.qTableChecksumPath(profile), 'w') as f:
            f.write(chksum)

    def loadQTable(self, profile: str) -> dict[Any, Any]:
        path = self.qTablePath(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        chkPath = self.qTableChecksumPath(profile)
        if os.path.exists(chkPath):
            try:
                with open(chkPath, 'r') as f:
                    saved = f.read()
                now = self._fileChecksum(path)
                if saved != now:
                    return {}
            except OSError:
                return {}
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError):
            return {}

    # ---------- Rewards log ----------
    def appendReward(self, profile: str, reward: float) -> None:
        d = self._profileDir(profile)
        self._ensureDir(d)
        with open(os.path.join(d, "SimulationRewards.txt"), 'a') as f:
            f.write(f"{reward}\n")

    # ---------- Profile dictionary helpers ----------
    def _readProfileDict(self, profile: str) -> ProfileDict:
        path = self._statsPath(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return self._migrateCountersFromLegacyProfile(profile)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return cast(ProfileDict, data)
                return {}
        except (pickle.UnpicklingError, EOFError, OSError):
            return {}

    def _migrateCountersFromLegacyProfile(self, profile: str) -> ProfileDict:
        """One-time migration: extract counter fields from profile.pkl into stats.pkl."""
        try:
            legacy = self._profilePath(profile)
            if not os.path.exists(legacy) or os.path.getsize(legacy) == 0:
                return {}
            with open(legacy, 'rb') as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                return {}
            rawDict = cast(ProfileDict, raw)
            counters: ProfileDict = {k: v for k, v in rawDict.items() if k in _STATS_KEYS}
            if counters:
                self._writeProfileDict(profile, counters)
            return counters
        except (pickle.UnpicklingError, EOFError, OSError):
            return {}

    def _writeProfileDict(self, profile: str, data: ProfileDict) -> None:
        d = self._profileDir(profile)
        self._ensureDir(d)
        path = self._statsPath(profile)
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='wb') as tmp:
            pickle.dump(data, tmp)
            temp = tmp.name
        self._atomicReplace(temp, path)

    def readProfileStats(self, profile: str) -> ProfileDict:
        return self._readProfileDict(profile)

    def updateProfileCounters(
        self,
        profile: str,
        *,
        totalSteps: int | None = None,
        nonRepeatingStepsTaken: int | None = None,
        timesRevisitedSquares: int | None = None,
        timesHitWallIncrement: int | None = None,
        timesHitEnemyIncrement: int | None = None,
    ) -> None:
        data = self._readProfileDict(profile)
        if totalSteps is not None:
            data['total_steps'] = int(data.get('total_steps', 0)) + int(totalSteps)
        if nonRepeatingStepsTaken is not None:
            data['non_repeating_steps_taken'] = int(data.get('non_repeating_steps_taken', 0)) + int(nonRepeatingStepsTaken)
        if timesRevisitedSquares is not None:
            data['times_revisited_squares'] = int(data.get('times_revisited_squares', 0)) + int(timesRevisitedSquares)
        if timesHitWallIncrement is not None:
            data['times_hit_wall'] = int(data.get('times_hit_wall', 0)) + int(timesHitWallIncrement)
        if timesHitEnemyIncrement is not None:
            data['times_hit_enemy'] = int(data.get('times_hit_enemy', 0)) + int(timesHitEnemyIncrement)
        self._writeProfileDict(profile, data)

    def incrementTimesHitWall(self, profile: str, n: int = 1) -> None:
        self.updateProfileCounters(profile, timesHitWallIncrement=int(n))

    def incrementTimesHitEnemy(self, profile: str, n: int = 1) -> None:
        self.updateProfileCounters(profile, timesHitEnemyIncrement=int(n))

    # ---------- Mazes.json helpers ----------
    def loadMazeData(self, profile: str) -> ProfileDict:
        import json, ast
        path = self.mazesJsonPath(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {
                "latest": {},
                "highest": {"reward": float('-inf')},
                "lowest": {"reward": float('inf')},
            }
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
                data: ProfileDict
                if isinstance(loaded, dict):
                    data = cast(ProfileDict, loaded)
                else:
                    data = {}
        except (json.JSONDecodeError, ValueError, OSError):
            return {
                "latest": {},
                "highest": {"reward": float('-inf')},
                "lowest": {"reward": float('inf')},
            }
        # Convert heatmap string keys to tuple keys for consumers
        for key in ('latest', 'highest', 'lowest'):
            container = data.get(key)
            if not isinstance(container, dict):
                continue
            containerDict = cast(dict[str, Any], container)
            hm = containerDict.get('heatmap_data')
            if not isinstance(hm, dict):
                continue
            hmDict = cast(dict[Any, Any], hm)
            converted: HeatmapData = {}
            for k, v in hmDict.items():
                tup: tuple[int, int] | None = None
                try:
                    obj = ast.literal_eval(str(k))
                    if isinstance(obj, (list, tuple)):
                        seq = cast(list[Any] | tuple[Any, ...], obj)
                        if len(seq) == 2:
                            tup = (int(seq[0]), int(seq[1]))
                except (ValueError, SyntaxError):
                    pass
                if tup is None:
                    try:
                        s = str(k).strip()
                        if s.startswith('(') and s.endswith(')'):
                            s = s[1:-1]
                        a, b = s.split(',', 1)
                        tup = (int(a.strip()), int(b.strip()))
                    except (ValueError, AttributeError):
                        continue
                converted[tup] = int(v)
            containerDict['heatmap_data'] = converted
            data[key] = containerDict
        return data

    def saveMazeEpisode(self, profile: str, maze: Any, heatmapData: HeatmapData, reward: float) -> None:
        import json
        d = self._profileDir(profile)
        self._ensureDir(d)
        path = self.mazesJsonPath(profile)
        data = self.loadMazeData(profile)
        # Store heatmap with string keys for JSON
        heatmapStr = {str(k): int(v) for k, v in heatmapData.items()}
        latest = {
            "maze": getattr(maze, 'grid', None),
            "start": getattr(maze, 'start', None),
            "end": getattr(maze, 'end', None),
            "heatmap_data": heatmapStr,
            "reward": float(reward),
        }
        data['latest'] = latest
        # Update bests with independent snapshots (avoid aliasing to latest)
        if float(reward) > float(data.get('highest', {}).get('reward', float('-inf'))):
            data['highest'] = dict(latest)
            highest = cast(dict[str, Any], data['highest'])
            highest['heatmap_data'] = dict(heatmapStr)
        if float(reward) < float(data.get('lowest', {}).get('reward', float('inf'))):
            data['lowest'] = dict(latest)
            lowest = cast(dict[str, Any], data['lowest'])
            lowest['heatmap_data'] = dict(heatmapStr)
        # Ensure existing snapshots are JSON-serializable (string keys)
        for key in ('latest', 'highest', 'lowest'):
            snapRaw = data.get(key)
            snap: dict[str, Any]
            if isinstance(snapRaw, dict):
                snap = cast(dict[str, Any], snapRaw)
            else:
                snap = {}
            hm = snap.get('heatmap_data')
            if isinstance(hm, dict):
                hmDict = cast(dict[Any, Any], hm)
                # Convert any tuple keys back to strings
                if any(isinstance(k, tuple) for k in hmDict.keys()):
                    snap['heatmap_data'] = {str(k): int(v) for k, v in hmDict.items()}
                else:
                    # normalize values to int for consistency
                    snap['heatmap_data'] = {str(k): int(v) for k, v in hmDict.items()}
            data[key] = snap
        # Atomic write
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='w') as tmp:
            json.dump(data, tmp, indent=4)
            temp = tmp.name
        self._atomicReplace(temp, path)

    def ensureProfileArtifacts(self, profile: str) -> None:
        """Create empty artifact stubs for a newly created profile."""
        d = self._profileDir(profile)
        self._ensureDir(d)
        for name in ("SimulationRewards.txt", "HeatmapData.txt"):
            p = os.path.join(d, name)
            if not os.path.exists(p):
                with open(p, 'w') as f:
                    f.write("")
        self.ensureMazeFile(profile)

    def ensureMazeFile(self, profile: str) -> None:
        """Create mazes.json with default structure if missing."""
        import json
        d = self._profileDir(profile)
        self._ensureDir(d)
        path = self.mazesJsonPath(profile)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        data = {
            "latest": {},
            "highest": {"reward": float('-inf')},
            "lowest": {"reward": float('inf')},
        }
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='w') as tmp:
            json.dump(data, tmp, indent=4)
            temp = tmp.name
        self._atomicReplace(temp, path)

    def clearProfileTrainingArtifacts(self, profile: str) -> None:
        """Delete durable training artifacts while keeping the profile itself."""
        pathsToRemove = (
            self.qTablePath(profile),
            self.qTableChecksumPath(profile),
            self.mazesJsonPath(profile),
            self.modelArtifactsDir(profile),
            os.path.join(self._profileDir(profile), "HeatmapData.txt"),
            self._statsPath(profile),
        )
        for path in pathsToRemove:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        rewardsPath = self.rewardsPath(profile)
        self._ensureDir(self._profileDir(profile))
        with open(rewardsPath, 'w') as f:
            f.write("")
        with open(os.path.join(self._profileDir(profile), "HeatmapData.txt"), 'w') as f:
            f.write("")
        self.ensureMazeFile(profile)

    def updateStepsFromHeatmap(self, profile: str, heatmapData: HeatmapData) -> None:
        """Update total_steps, times_revisited_squares, non_repeating_steps_taken from heatmap."""
        total = sum(int(v) for v in (heatmapData or {}).values())
        unique = len(heatmapData or {})
        repeated = max(0, total - unique)
        self.updateProfileCounters(
            profile,
            totalSteps=total,
            nonRepeatingStepsTaken=unique,
            timesRevisitedSquares=repeated,
        )
