"""一次性建池化 token 缓存（data*/pooled），避免多个训练作业各自从 NFS 读完整 token。  python -I build_cache.py demo,astra"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P
print('cached samples', len(P.load(sys.argv[1].split(','))), flush=True)
