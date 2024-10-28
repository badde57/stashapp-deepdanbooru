import os
import sys
import json
import sqlite3

METHOD = 'deepdanbooru-1.0.1'

try:
    import stashapi.log as log
    import stashapi.marker_parse as mp
    from stashapi.stashapp import StashInterface

    import os
    import sys
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    sys.path.append(os.path.join(sys.path[0],'tagger'))
    from tagger.tagger import Tagger, DEFAULTS
    from tqdm import tqdm

except ModuleNotFoundError:
    print("You need to install the stashapp-tools (stashapi) python module. (CLI: pip install stashapp-tools)", file=sys.stderr)

try:
    log.info(f"Available providers: {ort.get_available_providers()}")

    app = Tagger(
        model='wd-v1-4-moat-tagger.v2',
        embedder='gte',
        execution='cuda',
        threshold=0.35,
        max_tags=50,
        rawtag=True
    )
except Exception as e:
    log.error(f"Error initializing Deepdanbooru model: {e}")
    raise

# plugins don't start in the right directory, let's switch to the local directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def exit_plugin(msg=None, err=None):
    if msg is None and err is None:
        msg = "plugin ended"
    output_json = {"output": msg, "error": err}
    print(json.dumps(output_json))
    sys.exit()

def catchup():
    #f = {"stash_ids": {"modifier": "NOT_NULL"}}

    f = {
            "stash_id_endpoint": {
                "modifier": "NOT_NULL",
            }
        }
#                "stash_id": {"modifier": "NOT_NULL"}
    log.info('Getting scene count.')
    count=stash.find_scenes(f=f,filter={},get_count=True)[0]

    log.info(str(count)+' scenes to extract tags.')
    i=0
    for r in range(1,count+1):
        #log.info('fetching data: %s - %s %0.1f%%' % ((r - 1) * 1,r,(i/count)*100,))
        scenes=stash.find_scenes(f=f,filter={"page":r, "per_page": 1})
#        scenes=stash.find_scenes(f=f,filter={"page":r, "per_page": 1, "sort": "duration", "direction": "ASC"})
#        scenes=stash.find_scenes(f=f,filter={"page":r, "per_page": 1, "sort": "title", "direction": "ASC"})

        for s in scenes:
            if "stash_ids" not in s.keys():
                log.error(f"Scene {s['id']} must have stash_id, skipping...")
                continue
            elif len(s['files']) != 1:
                log.error(f"Scene {s['id']} must have exactly one file, skipping...")
                continue

            result = checktags(s)
            i=i+1
            log.progress((i/count))

def checktags(scene):
    file = scene['files'][0]
    scene_id = scene['id']
    path = file['path']
    file_id = file['id']
    fps = float(file['frame_rate'])
    dur = float(file['duration'])
    total_frames = int(dur * fps)
    #log.debug(f'processing {scene_id=}...')
    endpoint = scene['stash_ids'][0]['endpoint']
    stash_id = scene['stash_ids'][0]['stash_id']

    cur = con.cursor()
    cur.execute("SELECT 1 FROM deepdanbooru WHERE endpoint = ? AND stash_id = ?",(endpoint, stash_id,))
    rows = cur.fetchall()
    if len(rows) > 0:
        log.info(f"deepdanbooru - skipping {scene_id=}, already processed")
        return

    process_video(path, endpoint, stash_id)
    log.debug(f"deepdanbooru - finished {scene_id=}")
    return con.commit()

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_video(video_path, endpoint, stash_id, frequency=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frequency)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0

    cur = con.cursor()

    # Create a tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video", disable=True) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            if frame_count % frame_interval == 0:
                try:
                    time_offset = round(frame_count / fps, 2)

                    # Analyze the frame
                    ddb = app.process_image(pil_image)

                    tags = json.dumps(ddb['tags'])
                    ratings = json.dumps(ddb['ratings'])
                    embedding = json.dumps(ddb['embedding'].tolist() if isinstance(ddb['embedding'], np.ndarray) else ddb['embedding'])
                    #log.info(f"{stash_id=} {time_offset=} {','.join(ddb['tags'].keys())}")
                    cur.execute('INSERT INTO deepdanbooru (endpoint, stash_id, time_offset, tags, ratings, embedding, method) VALUES (?,?,?,?,?,?,?)',
                            (endpoint, stash_id, time_offset, tags, ratings, embedding, METHOD,)
                            )

                except Exception as e:
                    log.error(f"Error processing frame {frame_count}: {e}")

            frame_count += 1
            pbar.update(1)  # Update the progress bar

    cap.release()
    return con.commit()

def main():
    global stash
    json_input = json.loads(sys.stdin.read())
    FRAGMENT_SERVER = json_input["server_connection"]

    #log.debug(FRAGMENT_SERVER)

    stash = StashInterface(FRAGMENT_SERVER)
    PLUGIN_ARGS = False
    HOOKCONTEXT = False

    global con
    ddb_db_path = sys.argv[1]
    log.info(f"{ddb_db_path=}")
    con = sqlite3.connect(ddb_db_path)

    try:
#        PLUGIN_ARGS = json_input['args'].get("mode")
#        PLUGIN_DIR = json_input["PluginDir"]
        PLUGIN_ARGS = json_input['args']["mode"]
    except:
        pass

    if PLUGIN_ARGS:
        log.debug("--Starting Plugin 'deepdanbooru'--")
        if "catchup" in PLUGIN_ARGS:
            log.info("Catching up with deepdanbooru extraction on older files")
            catchup() #loops thru all scenes, and tag
        exit_plugin("deepdanbooru plugin finished")

    try:
        HOOKCONTEXT = json_input['args']["hookContext"]
    except:
        exit_plugin("deepdanbooru hook: No hook context")

    log.debug("--Starting Hook 'deepdanbooru'--")


    sceneID = HOOKCONTEXT['id']
    scene = stash.find_scene(sceneID)

    results = checktags(scene)
    con.close()
    exit_plugin(results)

main()
