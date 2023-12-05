# Makefile for PitchType project

# Default target
all: model ai

# Target for running the model training and evaluation
model:
	@echo "Running model training and evaluation..."
	python DraftModel.py

# Target for running the interactive AI for pitch prediction
ai:
	@echo "Running AI for pitch prediction..."
	python DraftAI.py

# Clean up joblib files
clean:
	@echo "Cleaning up joblib files..."
	rm -f *.joblib
	@echo "Cleanup complete."

.PHONY: all model ai clean
