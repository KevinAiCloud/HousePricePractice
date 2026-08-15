import os
import urllib.request
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

url = "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin"
output_path = "models/blip-image-captioning-base/pytorch_model.bin"
parts_dir = "models/blip-image-captioning-base/parts"

os.makedirs(parts_dir, exist_ok=True)

# 1. Fetch total size
print("Fetching total file size...")
try:
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes=0-0")
    with urllib.request.urlopen(req, timeout=30) as res:
        content_range = res.info().get("Content-Range", "")
        if "/" in content_range:
            total_size = int(content_range.split("/")[-1])
        else:
            total_size = 989820849
except Exception as e:
    print(f"Failed Content-Range: {e}. Defaulting.")
    total_size = 989820849

print(f"Total size: {total_size / (1024*1024):.2f} MB")

# 2. Divide file into 10MB chunks
CHUNK_SIZE = 10 * 1024 * 1024
chunks = []
start = 0
chunk_idx = 0
while start < total_size:
    end = min(start + CHUNK_SIZE - 1, total_size - 1)
    chunks.append((chunk_idx, start, end))
    start += CHUNK_SIZE
    chunk_idx += 1

total_chunks = len(chunks)
print(f"Divided file into {total_chunks} chunks.")

# Lock for printing progress safely
print_lock = Lock()
downloaded_bytes = 0

def download_chunk(item):
    global downloaded_bytes
    idx, start_byte, end_byte = item
    part_path = os.path.join(parts_dir, f"part_{idx}.tmp")
    expected_size = end_byte - start_byte + 1
    
    # Resume check: if part file already exists and is fully downloaded, skip
    if os.path.exists(part_path) and os.path.getsize(part_path) == expected_size:
        with print_lock:
            downloaded_bytes += expected_size
            percent = (downloaded_bytes / total_size) * 100
            sys.stdout.write(f"\rProgress: {downloaded_bytes/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({percent:.2f}%)")
            sys.stdout.flush()
        return idx, True
        
    retries = 5
    block_size = 64 * 1024
    while retries > 0:
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start_byte}-{end_byte}")
            with urllib.request.urlopen(req, timeout=30) as res:
                with open(part_path, "wb") as f:
                    downloaded = 0
                    while downloaded < expected_size:
                        limit = min(block_size, expected_size - downloaded)
                        block = res.read(limit)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        f.flush()
                        
                        with print_lock:
                            downloaded_bytes += len(block)
                            percent = (downloaded_bytes / total_size) * 100
                            sys.stdout.write(f"\rProgress: {downloaded_bytes/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({percent:.2f}%)")
                            sys.stdout.flush()
                if downloaded == expected_size:
                    return idx, True
                else:
                    raise ValueError(f"Incomplete read: got {downloaded} bytes, expected {expected_size}")
        except Exception as e:
            retries -= 1
            time.sleep(2)
            
    return idx, False

# 3. Download chunks in parallel
# Use 6 threads to balance download speed and connection limits
MAX_WORKERS = 6
print(f"Launching parallel downloads with {MAX_WORKERS} workers...")
failed = False

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_chunk, chunk): chunk for chunk in chunks}
    for future in as_completed(futures):
        idx, success = future.result()
        if not success:
            print(f"\nChunk {idx} failed to download.")
            failed = True

if failed:
    print("\nSome chunks failed to download. Run this script again to resume.")
    sys.exit(1)

# 4. Concatenate part files in order
print("\nAll chunks downloaded successfully. Assembling pytorch_model.bin...")
try:
    with open(output_path, "wb") as outfile:
        for idx in range(total_chunks):
            part_path = os.path.join(parts_dir, f"part_{idx}.tmp")
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
            # Cleanup temp part file
            os.remove(part_path)
            
    # Remove parts directory
    os.rmdir(parts_dir)
    print("Assembly complete! Model weights are ready.")
except Exception as e:
    print(f"\nAssembly failed: {e}")
    sys.exit(1)
