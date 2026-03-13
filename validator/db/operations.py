import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from ..challenge.challenge_types import (
    ChallengeType,
    GSRChallenge,
    GSRResponse,
    ValidationResult
)
from .schema import check_db_initialized, init_db

from fiber.logging_utils import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

        # Initialize database if needed
        if not check_db_initialized(str(db_path)):
            logger.info(f"Initializing new database at {db_path}")
            init_db(str(db_path))

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database at {db_path}")

    def close(self):
        if self.conn:
            self.conn.close()

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def store_challenge(self, challenge_id: int, challenge_type: str, video_url: str, task_name: str = None) -> None:
        """Store a new challenge in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO challenges (
                    challenge_id, type, video_url, created_at, task_name
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (
                challenge_id,
                challenge_type,
                video_url,
                task_name
            ))

            if cursor.rowcount == 0:
                logger.debug(f"Challenge {challenge_id} already exists in database")
            else:
                logger.info(f"Stored new challenge {challenge_id} in database")

            conn.commit()

        finally:
            conn.close()

    def assign_challenge(self, challenge_id: str, miner_hotkey: str, node_id: int) -> None:
        """Assign a challenge to a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO challenge_assignments (
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    status
                )
                VALUES (?, ?, ?, 'assigned')
            """, (
                challenge_id,
                miner_hotkey,
                node_id
            ))

            conn.commit()

        finally:
            conn.close()

    def mark_challenge_sent(self, challenge_id: str, miner_hotkey: str) -> None:
        """Mark a challenge as sent to a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'sent', sent_at = CURRENT_TIMESTAMP
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))

            conn.commit()

        finally:
            conn.close()

    def mark_challenge_failed(self, challenge_id: str, miner_hotkey: str) -> None:
        """Mark a challenge as failed for a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'failed'
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))

            conn.commit()

        finally:
            conn.close()

    def store_response(
        self,
        challenge_id: str,
        miner_hotkey: str,
        response: GSRResponse,
        node_id: int,
        processing_time: float = None,
        received_at: datetime = None,
        completed_at: datetime = None
    ) -> int:
        """Store a miner's response to a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.utcnow()

            # Convert response to dict and handle frames data
            response_dict = response.to_dict()

            # Store response
            cursor.execute("""
                INSERT INTO responses (
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    processing_time,
                    response_data,
                    received_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                challenge_id,
                miner_hotkey,
                node_id,
                processing_time,
                json.dumps(response_dict),
                received_at,
                completed_at
            ))

            response_id = cursor.lastrowid

            # Mark challenge as completed in challenge_assignments
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'completed',
                    completed_at = ?
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (now, challenge_id, miner_hotkey))

            conn.commit()
            return response_id

        finally:
            conn.close()

    def store_response_score(
        self,
        response_id: int,
        challenge_id: str,
        validation_result: ValidationResult,
        validator_hotkey: str,
        miner_hotkey: str,
        node_id: int,
        speed_score: float,
        total_score: float
    ) -> None:
        """Store the evaluation result for a response"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO response_scores (
                    response_id, challenge_id, evaluation_score, validator_hotkey,
                    miner_hotkey, node_id, availability_score, speed_score, total_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                response_id,
                challenge_id,
                validation_result.score,
                validator_hotkey,
                miner_hotkey,
                node_id,
                0.0,
                speed_score,
                total_score,
                datetime.utcnow()
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error storing response score: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_frame_evaluation(
        self,
        response_id: int,
        challenge_id: str,
        miner_hotkey: str,
        node_id: int,
        frame_id: int,
        frame_timestamp: float,
        frame_score: float,
        raw_frame_path: str,
        annotated_frame_path: str,
        vlm_response: dict,
        feedback: str
    ) -> None:
        """Store a frame evaluation result"""
        if response_id is None:
            raise ValueError("response_id is required for frame evaluation")

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO frame_evaluations (
                    response_id,
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    frame_id,
                    frame_timestamp,
                    frame_score,
                    raw_frame_path,
                    annotated_frame_path,
                    vlm_response,
                    feedback,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                response_id,
                challenge_id,
                miner_hotkey,
                node_id,
                frame_id,
                frame_timestamp,
                frame_score,
                raw_frame_path,
                annotated_frame_path,
                json.dumps(vlm_response),
                feedback
            ))

            conn.commit()
            logger.info(f"Stored frame evaluation for response {response_id}, frame {frame_id}")

        except Exception as e:
            logger.error(f"Error storing frame evaluation: {str(e)}")
            raise
        finally:
            conn.close()

    def get_miner_scores(self) -> Dict[int, Dict[str, Any]]:
        """Get calculated scores for miners from the last 72 hours"""
        query = """
        SELECT
            r.node_id,
            r.miner_hotkey,
            AVG(rs.evaluation_score) as performance_score,
            AVG(rs.speed_score) as speed_score,
            AVG(r.processing_time) as avg_processing_time,
            COUNT(*) as response_count,
            AVG(rs.total_score) as final_score,
            MAX(r.received_at) as last_active
        FROM responses r
        JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE r.received_at >= datetime('now', '-72 hours')
        GROUP BY r.node_id, r.miner_hotkey
        HAVING response_count > 0
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            miner_scores = {}
            for row in rows:
                node_id = int(row[0])
                more_recent_entry_already_exists = node_id in miner_scores and (
                    row[7] < miner_scores[node_id]['last_active']
                )
                if more_recent_entry_already_exists:
                    continue
                miner_scores[node_id] = {
                    'miner_hotkey': row[1],
                    'performance_score': row[2],
                    'speed_score': row[3],
                    'avg_processing_time': row[4],
                    'response_count': row[5],
                    'final_score': row[6],
                    'last_active': row[7]
                }

            logger.info(f"Fetched scores for {len(miner_scores)} miners")
            return miner_scores

    def get_miner_scores_with_node_id(self) -> Dict[int, Dict[str, Any]]:
        """Get calculated scores for miners from the last 72 hours, including node_id"""
        query = """
        SELECT
            r.node_id,
            r.miner_hotkey,
            AVG(rs.evaluation_score) as performance_score,
            AVG(rs.speed_score) as speed_score,
            AVG(r.processing_time) as avg_processing_time,
            COUNT(*) as response_count,
            AVG(rs.total_score) as final_score,
            MAX(r.received_at) as last_active
        FROM responses r
        JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE r.received_at >= datetime('now', '-72 hours')
        GROUP BY r.node_id, r.miner_hotkey
        HAVING response_count > 0
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            miner_scores = {}
            for row in rows:
                node_id = int(row[0])
                more_recent_entry_already_exists = node_id in miner_scores and (
                    row[7] < miner_scores[node_id]['last_active']
                )
                if more_recent_entry_already_exists:
                    continue
                miner_scores[node_id] = {
                    'miner_hotkey': row[1],
                    'performance_score': row[2],
                    'speed_score': row[3],
                    'avg_processing_time': row[4],
                    'response_count': row[5],
                    'final_score': row[6],
                    'last_active': row[7]
                }

            return miner_scores

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Get a challenge from the database by ID"""
        conn = self.get_connection()
        with conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, ca.sent_at
                FROM challenges c
                LEFT JOIN challenge_assignments ca ON c.challenge_id = ca.challenge_id
                WHERE c.challenge_id = ?
            """, (challenge_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None


    def get_frame_evaluations(
        self,
        challenge_id: str = None,
        miner_hotkey: str = None,
        node_id: int = None,
        response_id: int = None
    ) -> List[Dict]:
        """Get frame evaluations with optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            sql = "SELECT * FROM frame_evaluations WHERE 1=1"
            params = []

            if challenge_id:
                sql += " AND challenge_id = ?"
                params.append(challenge_id)
            if miner_hotkey:
                sql += " AND miner_hotkey = ?"
                params.append(miner_hotkey)
            if node_id:
                sql += " AND node_id = ?"
                params.append(node_id)
            if response_id:
                sql += " AND response_id = ?"
                params.append(response_id)

            sql += " ORDER BY frame_timestamp ASC"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                result = dict(row)
                if result['vlm_response']:
                    result['vlm_response'] = json.loads(result['vlm_response'])
                results.append(result)

            return results

        finally:
            conn.close()

    def get_processing_time_stats(self, challenge_id: str) -> Dict[str, float]:
        """
        Get processing time statistics for all responses to the same challenge.
        Processing times are in seconds.

        Args:
            challenge_id: The challenge ID to get stats for

        Returns:
            Dict with avg_time, min_time, max_time in seconds
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    AVG(processing_time) as avg_time,
                    MIN(processing_time) as min_time,
                    MAX(processing_time) as max_time
                FROM responses
                WHERE challenge_id = ?
                AND processing_time > 0
            """, (str(challenge_id),))

            row = cursor.fetchone()
            if row:
                return {
                    'avg_time': row[0] or 100.0,  # Default to 5 seconds if no data
                    'min_time': row[1] or 5.0,  # Minimum 1 second
                    'max_time': row[2] or 200.0  # Maximum 10 seconds
                }
            return {
                'avg_time': 100.0,  # Default values in seconds
                'min_time': 5.0,
                'max_time': 200.0
            }

        finally:
            conn.close()

    def get_completed_tasks(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get completed tasks from the last N hours"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    ca.challenge_id,
                    ca.node_id,
                    ca.miner_hotkey,
                    ca.sent_at,
                    ca.received_at,
                    ca.task_returned_data,
                    c.type,
                    c.task_name
                FROM challenge_assignments ca
                JOIN challenges c ON ca.challenge_id = c.challenge_id
                WHERE ca.status = 'completed'
                AND ca.received_at >= datetime('now', ? || ' hours')
                ORDER BY ca.received_at DESC
            """, (-hours,))

            rows = cursor.fetchall()
            tasks = []
            for row in rows:
                task = {
                    'task_id': row[0],
                    'node_id': row[1],
                    'miner_hotkey': row[2],
                    'sent_at': row[3],
                    'received_at': row[4],
                    'task_returned_data': row[5],
                    'type': row[6],
                    'task_name': row[7]
                }
                tasks.append(task)

            return tasks

        finally:
            conn.close()

    def get_challenges_with_unevaluated_responses(self) -> List[Dict]:
        """Get challenges that have responses without evaluations"""
        query = """
        SELECT DISTINCT c.*
        FROM challenges c
        JOIN responses r ON c.challenge_id = r.challenge_id
        LEFT JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE rs.response_id IS NULL
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_unevaluated_responses(self, challenge_id: str) -> List[Dict]:
        """Get responses for a challenge that haven't been evaluated yet"""
        query = """
        SELECT
            r.response_id,
            r.challenge_id,
            r.node_id,
            r.miner_hotkey,
            r.processing_time,
            r.response_data
        FROM responses r
        LEFT JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE r.challenge_id = ? AND rs.response_id IS NULL
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (challenge_id,))
            rows = cursor.fetchall()

            responses = []
            for row in rows:
                # Create GSRResponse compatible dict
                gsr_response = {
                    'challenge_id': row[1],
                    'node_id': row[2],
                    'miner_hotkey': row[3],
                    'processing_time': row[4],
                    'frames': {}
                }

                # Parse response_data JSON and extract frames
                if row[5]:  # response_data
                    try:
                        response_data = json.loads(row[5])
                        # The frames data is nested inside response_data['frames']
                        frames_data = response_data.get('frames', {})
                        if isinstance(frames_data, dict):
                            gsr_response['frames'] = frames_data
                        else:
                            logger.warning(f"Frames data in response {row[0]} is not a dictionary: {type(frames_data)}")
                            gsr_response['frames'] = {}
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse response data for response {row[0]}: {e}")
                        gsr_response['frames'] = {}

                # Add response_id as a separate field
                gsr_response['response_id'] = row[0]
                responses.append(gsr_response)

            return responses

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Get challenge details"""
        conn = self.get_connection()
        with conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, ca.sent_at
                FROM challenges c
                LEFT JOIN challenge_assignments ca ON c.challenge_id = ca.challenge_id
                WHERE c.challenge_id = ?
            """, (challenge_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_challenge_frames(self, challenge_id: str) -> List[int]:
        """Get frame numbers selected for a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT frame_number
                FROM challenge_frames
                WHERE challenge_id = ?
                ORDER BY frame_number
            """, (challenge_id,))

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def store_challenge_frames(self, challenge_id: str, frame_numbers: List[int]) -> None:
        """Store selected frame numbers for a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.executemany("""
                INSERT INTO challenge_frames (challenge_id, frame_number)
                VALUES (?, ?)
            """, [(challenge_id, frame_num) for frame_num in frame_numbers])

            conn.commit()

        finally:
            conn.close()

    def get_frame_scores(self, challenge_id: str, response_id: int) -> List[float]:
        """Get all frame scores for a response"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT frame_score
                FROM frame_evaluations
                WHERE challenge_id = ? AND response_id = ?
                ORDER BY frame_id
            """, (challenge_id, response_id))

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def update_response_score(self, response_id: int, score: float) -> None:
        """Update the overall score for a response and mark it as evaluated"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE responses
                SET score = ?, evaluated = TRUE, evaluated_at = ?
                WHERE id = ?
            """, (score, datetime.utcnow(), response_id))

            conn.commit()

        finally:
            conn.close()



    async def create_challenge(self, video_url: str, external_task_id: int) -> Optional[int]:
        """
        Create a new challenge in the database.

        Args:
            video_url: URL of the video for the challenge
            external_task_id: Task ID from the external API

        Returns:
            challenge_id if successful, None otherwise
        """
        try:
            query = """
            INSERT INTO challenges (video_url, external_task_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            RETURNING challenge_id
            """

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (video_url, external_task_id))
                row = cursor.fetchone()
                conn.commit()

                if row:
                    challenge_id = row[0]
                    logger.info(f"Created challenge {challenge_id} for external task {external_task_id}")
                    return challenge_id

                return None

        except Exception as e:
            logger.error(f"Error creating challenge: {str(e)}")
            return None

    def has_challenge_assignment(self, challenge_id: str, miner_hotkey: str) -> bool:
        """Check if a miner has already been assigned a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT 1
                FROM challenge_assignments
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))

            return cursor.fetchone() is not None

        finally:
            conn.close()



    def cleanup_old_data(self, days: int = 7) -> None:
        """
        Delete everything older than x days.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
    
        try:
            # tables to clean by timestamp
            tables_to_clean = [
                ("responses", "received_at"),
                ("frame_evaluations", "created_at"),
                ("response_scores", "created_at"),
                ("availability_checks", "checked_at"),
            ]
    
            for table, timestamp_column in tables_to_clean:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE {timestamp_column} IS NOT NULL
                    AND {timestamp_column} < datetime('now', '-{days} days')
                """)
                logger.info(f"Deleted {cursor.rowcount} rows from {table} older than {days} days")
    
            # delete challenge assignement
            cursor.execute(f"""
                DELETE FROM challenge_assignments
                WHERE 
                    (sent_at IS NOT NULL AND sent_at < datetime('now', '-{days} days'))
                    OR
                    (completed_at IS NOT NULL AND completed_at < datetime('now', '-{days} days'))
            """)
            logger.info(f"Deleted {cursor.rowcount} old challenge_assignments older than {days} days")
    
            # clanup orphan challenges
            cursor.execute(f"""
                DELETE FROM challenges
                WHERE challenge_id NOT IN (
                    SELECT DISTINCT challenge_id FROM responses
                    UNION
                    SELECT DISTINCT challenge_id FROM challenge_assignments
                )
                AND created_at < datetime('now', '-{days} days')
            """)
            logger.info(f"Deleted {cursor.rowcount} orphaned challenges older than {days} days")
    
            conn.commit()
            logger.info(f"Database cleanup completed for data older than {days} days")
    
        except Exception as e:
            conn.rollback()
            logger.error(f"Error during database cleanup: {str(e)}")
        finally:
            conn.close()

    def mark_response_as_evaluated(self, response_id: int) -> None:
        """Mark a response as evaluated"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE responses
                SET evaluated = TRUE, evaluated_at = ?
                WHERE response_id = ?
            """, (datetime.utcnow(), response_id))

            conn.commit()

        finally:
            conn.close()

    def get_challenge_assignment_sent_at(self, challenge_id: str, miner_hotkey: str) -> Optional[datetime]:
        """Get the sent_at timestamp for a challenge assignment"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT sent_at
                FROM challenge_assignments
                WHERE challenge_id = ? AND miner_hotkey = ?
                AND status IN ('sent', 'completed')
            """, (str(challenge_id), miner_hotkey))
            row = cursor.fetchone()
            return datetime.fromisoformat(row[0]) if row and row[0] else None
        finally:
            conn.close()

    def update_response(
        self,
        response_id: int,
        score: float,
        evaluated: bool,
        evaluated_at: datetime
    ) -> None:
        """Update a response with evaluation results"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE responses
                SET score = ?, evaluated = ?, evaluated_at = ?
                WHERE response_id = ?
            """, (score, evaluated, evaluated_at, response_id))

            conn.commit()
            logger.info(f"Updated response {response_id} with score {score}")

        except Exception as e:
            logger.error(f"Error updating response {response_id}: {str(e)}")
            conn.rollback()
        finally:
            conn.close()

    def get_response_completed_at(self, response_id: str) -> Optional[datetime]:
        """
        Get the completed_at timestamp for a given response.
        """
        query = "SELECT completed_at FROM responses WHERE response_id = ?"
        result = self.execute_query(query, (response_id,), fetch_one=True)
        if result and result["completed_at"]:
            return datetime.fromisoformat(result["completed_at"])
        return None

    def execute_query(self, query: str, params: tuple = (), fetch_one: bool = False) -> Optional[Any]:
        """Execute a SQL query and return results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)

            if fetch_one:
                result = cursor.fetchone()
            else:
                result = cursor.fetchall()

            conn.commit()
            return result
        except Exception as e:
            logger.error(f"Database error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        finally:
            conn.close()

    def get_sample_responses(self, challenge_id: str, sample_size: int) -> List[Dict]:
        """
        Get a sample of responses for a given challenge.
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT response_id, response_data
                FROM responses
                WHERE challenge_id = ?
                  AND response_data IS NOT NULL
                ORDER BY RANDOM()
                LIMIT ?
            """, (challenge_id, sample_size))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            cursor.close()
            conn.close()

    async def get_pending_responses(self, challenge_id: str) -> List[GSRResponse]:
        """
        Get all unevaluated responses for a challenge.

        Args:
            challenge_id: The challenge ID to get responses for

        Returns:
            List of GSRResponse objects
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    response_id,
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    response_data,
                    processing_time,
                    received_at,
                    completed_at
                FROM responses
                WHERE challenge_id = ?
                  AND evaluated = FALSE
                  AND response_data IS NOT NULL
            """, (challenge_id,))

            rows = cursor.fetchall()
            responses = []

            for row in rows:
                try:
                    response_data = json.loads(row["response_data"]) if row["response_data"] else {}
                    response = GSRResponse(
                        challenge_id=row["challenge_id"],
                        miner_hotkey=row["miner_hotkey"],
                        frames=response_data.get("frames", {}),
                        processing_time=row["processing_time"],
                        response_id=row["response_id"],
                        node_id=row["node_id"]
                    )
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing response {row['response_id']}: {str(e)}")
                    continue

            logger.info(f"Found {len(responses)} pending responses for challenge {challenge_id}")
            return responses

        finally:
            cursor.close()
            conn.close()

    def mark_responses_failed(self, challenge_id):
        """Mark all responses for a challenge as evaluated if the video is missing."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE responses
                SET evaluated = TRUE, evaluated_at = ?
                WHERE challenge_id = ?
            """, (datetime.utcnow(), challenge_id))

            conn.commit()
            logger.info(f" All responses for challenge {challenge_id} marked as evaluated (skipped due to 404).")

        except Exception as e:
            conn.rollback()  #
            logger.error(f" Error updating responses for challenge {challenge_id}: {str(e)}")

        finally:
            cursor.close()
            conn.close()


    def mark_response_failed(self, response_id: int) -> None:
        """Mark a single response as evaluated (used when evaluation failed)."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE responses
                SET evaluated = TRUE, evaluated_at = ?
                WHERE response_id = ?
            """, (datetime.utcnow(), response_id))

            conn.commit()
            logger.info(f"Response {response_id} marked as evaluated (skipped due to error).")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error marking response {response_id} as failed: {str(e)}")

        finally:
            cursor.close()
            conn.close()
