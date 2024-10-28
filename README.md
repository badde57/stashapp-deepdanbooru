# stashapp-deepdanbooru - Deepdanbooru frame tagging Stashapp

## Purpose

Frame tagging.

## How to configure the plugin

0. Install requirements: `pip install -r requirements.txt`. At a high level,
   it's deepdanbooru, huggingface-hub, and onnxruntime-gpu.

   Tested with Python 3.10

1. Create a database for storing tags:
```
  echo "
    CREATE TABLE deepdanbooru (
      endpoint TEXT NOT NULL,
      stash_id TEXT NOT NULL,
      time_offset FLOAT NOT NULL,
      tags TEXT NOT NULL,
      ratings TEXT NOT NULL,
      embedding TEXT NOT NULL,
      method TEXT NOT NULL,
      UNIQUE (stash_id, method, time_offset)
    );
  " | sqlite3 /path/to/meta.sqlite
```
