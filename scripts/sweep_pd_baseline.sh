#!/bin/bash

BASEDIR=$(dirname "$0")
wandb sweep --project pdcontrol "$BASEDIR/sweep_pd_baseline.yaml"