import os
import pickle
import hashlib
import tempfile


class ArtifactsRepository:
    """Centralized persistence for training artifacts under profiles/<name>.

    Scope:
    - Q-table read/write (+ checksum)
    - Rewards log append
    - Maze runs (mazes.json) latest/highest/lowest
    - Profile aggregated stats (profile.pkl) counters

    All writes are atomic.
    """

    def __init__(self, base_dir: str = "profiles"):
        self.base_dir = base_dir

    # ---------- Helpers ----------
    def _profile_dir(self, profile: str) -> str:
        return os.path.join(self.base_dir, profile)

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    # ---------- Generic paths ----------
    def _profile_path(self, profile: str) -> str:
        return os.path.join(self._profile_dir(profile), "profile.pkl")

    def rewards_path(self, profile: str) -> str:
        return os.path.join(self._profile_dir(profile), "SimulationRewards.txt")

    def mazes_json_path(self, profile: str) -> str:
        return os.path.join(self._profile_dir(profile), "mazes.json")

    # ---------- Q-table ----------
    def q_table_path(self, profile: str) -> str:
        return os.path.join(self._profile_dir(profile), "q_table.pkl")

    def q_table_checksum_path(self, profile: str) -> str:
        return os.path.join(self._profile_dir(profile), "q_table.checksum")

    def _file_checksum(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def save_q_table(self, profile: str, q_table: dict) -> None:
        d = self._profile_dir(profile)
        self._ensure_dir(d)
        path = self.q_table_path(profile)
        # Atomic write
        with tempfile.NamedTemporaryFile(delete=False, dir=d) as tmp:
            pickle.dump(q_table, tmp)
            temp_name = tmp.name
        os.replace(temp_name, path)
        # Write checksum
        chksum = self._file_checksum(path)
        with open(self.q_table_checksum_path(profile), 'w') as f:
            f.write(chksum)

    def load_q_table(self, profile: str) -> dict:
        path = self.q_table_path(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        chk_path = self.q_table_checksum_path(profile)
        if os.path.exists(chk_path):
            try:
                with open(chk_path, 'r') as f:
                    saved = f.read()
                now = self._file_checksum(path)
                if saved != now:
                    # Corrupt or partial file; ignore
                    return {}
            except Exception:
                return {}
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}

    # ---------- Rewards log ----------
    def append_reward(self, profile: str, reward: float) -> None:
        d = self._profile_dir(profile)
        self._ensure_dir(d)
        with open(os.path.join(d, "SimulationRewards.txt"), 'a') as f:
            f.write(f"{reward}\n")

    # ---------- Profile dictionary helpers ----------
    def _read_profile_dict(self, profile: str) -> dict:
        path = self._profile_path(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_profile_dict(self, profile: str, data: dict) -> None:
        d = self._profile_dir(profile)
        self._ensure_dir(d)
        path = self._profile_path(profile)
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='wb') as tmp:
            pickle.dump(data, tmp)
            temp = tmp.name
        os.replace(temp, path)

    def read_profile_stats(self, profile: str) -> dict:
        return self._read_profile_dict(profile)

    def update_profile_counters(
        self,
        profile: str,
        *,
        total_steps: int | None = None,
        non_repeating_steps_taken: int | None = None,
        times_revisited_squares: int | None = None,
        times_hit_wall_increment: int | None = None,
    ) -> None:
        data = self._read_profile_dict(profile)
        if total_steps is not None:
            data['total_steps'] = int(data.get('total_steps', 0)) + int(total_steps)
        if non_repeating_steps_taken is not None:
            data['non_repeating_steps_taken'] = int(data.get('non_repeating_steps_taken', 0)) + int(non_repeating_steps_taken)
        if times_revisited_squares is not None:
            data['times_revisited_squares'] = int(data.get('times_revisited_squares', 0)) + int(times_revisited_squares)
        if times_hit_wall_increment is not None:
            data['times_hit_wall'] = int(data.get('times_hit_wall', 0)) + int(times_hit_wall_increment)
        self._write_profile_dict(profile, data)

    def increment_times_hit_wall(self, profile: str, n: int = 1) -> None:
        self.update_profile_counters(profile, times_hit_wall_increment=int(n))

    # ---------- Mazes.json helpers ----------
    def load_maze_data(self, profile: str) -> dict:
        import json, ast
        path = self.mazes_json_path(profile)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {
                "latest": {},
                "highest": {"reward": float('-inf')},
                "lowest": {"reward": float('inf')},
            }
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            return {
                "latest": {},
                "highest": {"reward": float('-inf')},
                "lowest": {"reward": float('inf')},
            }
        # Convert heatmap string keys to tuple keys for consumers
        for key in ('latest', 'highest', 'lowest'):
            hm = (data.get(key) or {}).get('heatmap_data')
            if not isinstance(hm, dict):
                continue
            converted = {}
            for k, v in hm.items():
                tup = None
                try:
                    obj = ast.literal_eval(k)
                    if isinstance(obj, (list, tuple)) and len(obj) == 2:
                        tup = (int(obj[0]), int(obj[1]))
                except Exception:
                    pass
                if tup is None:
                    try:
                        s = str(k).strip()
                        if s.startswith('(') and s.endswith(')'):
                            s = s[1:-1]
                        a, b = s.split(',', 1)
                        tup = (int(a.strip()), int(b.strip()))
                    except Exception:
                        continue
                converted[tup] = v
            data[key]['heatmap_data'] = converted
        return data

    def save_maze_episode(self, profile: str, maze, heatmap_data: dict, reward: float) -> None:
        import json
        from copy import deepcopy
        d = self._profile_dir(profile)
        self._ensure_dir(d)
        path = self.mazes_json_path(profile)
        data = self.load_maze_data(profile)
        # Store heatmap with string keys for JSON
        heatmap_str = {str(k): int(v) for k, v in (heatmap_data or {}).items()}
        latest = {
            "maze": getattr(maze, 'grid', None),
            "start": getattr(maze, 'start', None),
            "end": getattr(maze, 'end', None),
            "heatmap_data": heatmap_str,
            "reward": float(reward),
        }
        data['latest'] = latest
        # Update bests with independent snapshots (avoid aliasing to latest)
        if float(reward) > float(data.get('highest', {}).get('reward', float('-inf'))):
            data['highest'] = dict(latest)
            data['highest']['heatmap_data'] = dict(latest['heatmap_data'])
        if float(reward) < float(data.get('lowest', {}).get('reward', float('inf'))):
            data['lowest'] = dict(latest)
            data['lowest']['heatmap_data'] = dict(latest['heatmap_data'])
        # Ensure existing snapshots are JSON-serializable (string keys)
        for key in ('latest', 'highest', 'lowest'):
            snap = data.get(key) or {}
            hm = snap.get('heatmap_data')
            if isinstance(hm, dict):
                # Convert any tuple keys back to strings
                if any(isinstance(k, tuple) for k in hm.keys()):
                    snap['heatmap_data'] = {str(k): int(v) for k, v in hm.items()}
                else:
                    # normalize values to int for consistency
                    snap['heatmap_data'] = {str(k): int(v) for k, v in hm.items()}
            data[key] = snap
        # Atomic write
        with tempfile.NamedTemporaryFile(delete=False, dir=d, mode='w') as tmp:
            json.dump(data, tmp, indent=4)
            temp = tmp.name
        os.replace(temp, path)

    def ensure_maze_file(self, profile: str) -> None:
        """Create mazes.json with default structure if missing."""
        import json
        d = self._profile_dir(profile)
        self._ensure_dir(d)
        path = self.mazes_json_path(profile)
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
        os.replace(temp, path)

    def update_steps_from_heatmap(self, profile: str, heatmap_data: dict) -> None:
        """Update total_steps, times_revisited_squares, non_repeating_steps_taken from heatmap."""
        total = int(sum(int(v) for v in (heatmap_data or {}).values()))
        unique = int(len(heatmap_data or {}))
        repeated = int(max(0, total - unique))
        self.update_profile_counters(
            profile,
            total_steps=total,
            non_repeating_steps_taken=unique,
            times_revisited_squares=repeated,
        )
