"""Spawn the S3 export (app maemm-export-s3-v2, data/modal_export_s3_v2.py). Reads the AWS default profile from ~/.aws/credentials at
call time and passes the keys as call arguments; nothing is printed or stored except the FunctionCall id
(-> ~/shared/overnight/simple2m/ids.json["s3_export"]). Run by the owner:

    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 ~/maemm-pub-simple2m/scripts/launchers/spawn_export_s3_v2.py
"""
import configparser
import json
import os
import sys
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
cp = configparser.ConfigParser(); cp.read(os.path.expanduser("~/.aws/credentials"))
key, secret = cp["default"]["aws_access_key_id"], cp["default"]["aws_secret_access_key"]
BUCKET = os.environ.get("ARB_BUCKET", "celeste-maemm-27b-data")
PREFIX = os.environ.get("ARB_PREFIX", time.strftime("v2-%Y-%m-%d", time.gmtime()))
legacy = "--no-legacy" not in sys.argv
fc = modal.Function.from_name("maemm-export-s3-v2", "export_s3").spawn(bucket=BUCKET, prefix=PREFIX, aws_key=key, aws_secret=secret, region="us-east-1",
                                                                       include_training=True, include_legacy=legacy, presign_days=7)
rec = {"call": fc.object_id, "bucket": BUCKET, "prefix": PREFIX, "include_legacy": legacy, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
p = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
d = json.load(open(p)) if os.path.exists(p) else {}
d["s3_export"] = rec; json.dump(d, open(p, "w"), indent=1)
print(json.dumps(rec))
print(f"poll: python3 -c \"import modal; print(modal.FunctionCall.from_id('{fc.object_id}').get(timeout=0))\"   (raises TimeoutError while running)")
