#!/bin/bash

BASEDIR=$(dirname "$0")
wandb sweep --project sb3-single-agent "$BASEDIR/sweep_rewardshaping.yaml"