import json
import random
import time
import uuid
import logging
from datetime import datetime

# Configure logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(event_type, **data):
    """
    Emit structured JSON logs for observability across the pipeline.
    Ensures every event has a standardized timestamp and metadata.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        **data
    }
    # Using print for raw log output as per original design
    print(json.dumps(payload))


class ProgressListener:
    """
    Emits throttled progress updates (~20% intervals).
    Tracks state for failure recovery and observability.
    """
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.last_bucket = 0
        self.last_step = 0

    def __call__(self, trainer_name, step, total_steps):
        """
        Calculates and logs progress buckets to reduce log noise.
        """
        self.last_step = step
        percent_complete = (step / total_steps) * 100
        bucket = int(percent_complete // 20)  # 20% intervals

        if bucket > self.last_bucket:
            self.last_bucket = bucket
            log_event(
                "trainer_progress",
                trace_id=self.trace_id,
                span_id=self.span_id,
                trainer=trainer_name,
                step=step,
                total_steps=total_steps,
                percent_complete=round(percent_complete, 2)
            )


class Trainer:
    """
    Simulated Trainer performing N training iterations.
    Includes failure injection for testing pipeline observability.
    """
    def __init__(self, name, iterations, fail_at_iter=None, learning_rate=0.01):
        self.name = name
        self.iterations = iterations
        self.fail_at_iter = fail_at_iter
        self.learning_rate = learning_rate

    def perform_training_step(self, step):
        """Internal logic for a single training step."""
        time.sleep(random.uniform(0.05, 0.15))
        if self.fail_at_iter and step == self.fail_at_iter:
            raise RuntimeError(f"Critical failure in {self.name} at iteration {step}")

    def run(self, listener):
        """Executes the training loop."""
        for step in range(1, self.iterations + 1):
            self.perform_training_step(step)
            listener(self.name, step, self.iterations)


class Orchestrator:
    """
    Runs Trainers sequentially while emitting observability events.
    Manages the full lifecycle of the training pipeline.
    """
    def __init__(self, trainers):
        self.trainers = trainers

    def run(self):
        trace_id = str(uuid.uuid4())
        total_pipeline_iters = sum(t.iterations for t in self.trainers)
        completed_pipeline_iters = 0
        run_start = time.time()

        log_event("pipeline_started", trace_id=trace_id, total_trainers=len(self.trainers))

        try:
            for trainer in self.trainers:
                span_id = str(uuid.uuid4())
                listener = ProgressListener(trace_id, span_id)
                start_time = time.time()

                log_event("trainer_started", trace_id=trace_id, span_id=span_id, trainer=trainer.name)

                try:
                    trainer.run(listener)
                    duration = time.time() - start_time
                    completed_pipeline_iters += trainer.iterations
                    pipeline_percent = (completed_pipeline_iters / total_pipeline_iters) * 100

                    log_event(
                        "trainer_completed",
                        trace_id=trace_id,
                        span_id=span_id,
                        trainer=trainer.name,
                        duration_seconds=round(duration, 3),
                        pipeline_percent=round(pipeline_percent, 2)
                    )
                except Exception as e:
                    # Failure handling logic
                    log_event(
                        "trainer_failed",
                        trace_id=trace_id,
                        span_id=span_id,
                        trainer=trainer.name,
                        error=str(e)
                    )
                    raise

            # Summary on success
            run_duration = time.time() - run_start
            log_event("run_summary", status="success", total_duration=round(run_duration, 3))

        except Exception:
            log_event("run_summary", status="failed", total_duration=round(time.time() - run_start, 3))
            print("\nPipeline failed gracefully.\n")


def main():
    """Main execution point for the training orchestration."""
    trainers = [
        Trainer("DataIngestor", 5),
        Trainer("FeatureExtractor", 10),
        Trainer("ModelTrainer", 8),
        Trainer("Evaluator", 4)
    ]

    orchestrator = Orchestrator(trainers)
    orchestrator.run()

if __name__ == "__main__":
    main()
