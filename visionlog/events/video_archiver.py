import os
import time
import threading
import uuid
import json
import logging
from pathlib import Path
import cv2

from visionlog import config

logger = logging.getLogger(__name__)

class VideoArchiver:
    """
    Background worker that saves frames into N-second MP4 chunks.
    Once a chunk is finished, it spawns a thread to upload it to Twelve Labs
    and index the embeddings into Pinecone (visionlog-video namespace: 'visionlog-video').
    """

    def __init__(self, camera_id: str, fps: float, width: int, height: int, is_live: bool = False):
        self.camera_id = camera_id
        self.fps = fps or 30.0
        self.width = width
        self.height = height
        self.is_live = is_live
        
        self.enabled = getattr(config, "ENABLE_VIDEO_INDEX", False)
        self.chunk_s = getattr(config, "VIDEO_CHUNK_S", 6)
        self.frames_per_chunk = int(self.fps * self.chunk_s)
        
        # Temp dir for video chunks
        self.tmp_dir = Path("tmp_video_chunks")
        self.tmp_dir.mkdir(exist_ok=True)
        
        # Current chunk state
        self.current_writer = None
        self.current_filepath = None
        self.current_frame_count = 0
        self.chunk_start_ts = None
        
        if self.enabled:
            print(f"[VideoArchiver] {camera_id} initialized (Chunk: {self.chunk_s}s, Live: {is_live})")
            
            # Init clients early to fail fast if keys are missing
            if not config.TWELVE_LABS_API_KEY or not config.PINECONE_API_KEY:
                print("[VideoArchiver] WARNING: Missing API keys. Video indexing disabled.")
                self.enabled = False

    def add_frame(self, frame, frame_ts: float):
        """Called every frame by the main loop."""
        if not self.enabled:
            return

        if self.current_writer is None:
            self._start_new_chunk(frame_ts)

        self.current_writer.write(frame)
        self.current_frame_count += 1
        self.last_ts = frame_ts

        if self.current_frame_count >= self.frames_per_chunk:
            self.flush()

    def _start_new_chunk(self, frame_ts: float):
        self.chunk_start_ts = frame_ts
        self.current_frame_count = 0
        chunk_id = f"{self.camera_id}_{int(frame_ts)}_{uuid.uuid4().hex[:6]}"
        self.current_filepath = self.tmp_dir / f"{chunk_id}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_writer = cv2.VideoWriter(
            str(self.current_filepath), fourcc, self.fps, (self.width, self.height)
        )

    def flush(self):
        """Close current chunk and spawn upload thread."""
        if not self.enabled or self.current_writer is None:
            return

        self.current_writer.release()
        self.current_writer = None
        
        end_ts = self.last_ts
        filepath = self.current_filepath
        start_ts = self.chunk_start_ts
        
        print(f"[VideoArchiver] Chunk ready: {filepath.name} ({self.current_frame_count} frames)")
        
        # Spawn background uploader
        t = threading.Thread(
            target=self._upload_worker, 
            args=(filepath, start_ts, end_ts),
            daemon=True
        )
        t.start()

    def _upload_worker(self, filepath: Path, start_ts: float, end_ts: float):
        """Uploads mp4 to TwelveLabs -> gets embedding -> upserts to Pinecone."""
        try:
            from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource
            from pinecone import Pinecone
            
            # 1. Init clients
            tl_client = TwelveLabs(api_key=config.TWELVE_LABS_API_KEY)
            pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pc_client.Index(config.PINECONE_VIDEO_INDEX)
            
            # Determine if this was live wall-clock time or relative file time
            # For Pinecone metadata, we want the absolute Unix timestamp that matches text events.
            # In live mode: start_ts and end_ts are already wall-clock time.time().
            # In file mode: start_ts is relative (0.0 to N). Our video_source.py already computes
            # an anchor recording_start + relative_pos. So frame_ts is already an absolute 
            # Unix timestamp matching the text events! This is perfect.
            
            # 2. Upload to Twelve Labs (Synchronous for short videos)
            try:
                # Use sync endpoint for <10min clips
                response = tl_client.embed.v_2.create(
                    input_type="video",
                    model_name="marengo3.0",
                    video=VideoInputRequest(
                        media_source=MediaSource(
                            method="direct",
                            file=open(str(filepath), "rb")
                        ),
                        embedding_option=["visual"],
                        embedding_scope=["asset"] # One embedding for the whole 6s chunk
                    )
                )
            except Exception as e:
                print(f"[VideoArchiver] Twelve Labs Error: {e}")
                return

            if not response.data:
                print(f"[VideoArchiver] No embeddings returned for {filepath.name}")
                return
                
            embedding_vector = response.data[0].embedding
            
            # 3. Upsert to Pinecone
            vector_id = f"{self.camera_id}_{int(start_ts)}_{uuid.uuid4().hex[:8]}"
            
            metadata = {
                "camera_name": self.camera_id, # Can join with rich name later
                "start_ts": start_ts,
                "end_ts": end_ts,
                "is_live": self.is_live,
                "duration_s": end_ts - start_ts
                # Could optionally upload to S3 here and store URL in metadata
            }

            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding_vector,
                    "metadata": metadata
                }]
            )
            
            print(f"[VideoArchiver] Upserted {vector_id} to {config.PINECONE_VIDEO_INDEX}!")

        except Exception as e:
            print(f"[VideoArchiver] Worker Error on {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 4. Clean up temporary mp4 file
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    print(f"[VideoArchiver] Warn: could not delete {filepath}: {e}")
