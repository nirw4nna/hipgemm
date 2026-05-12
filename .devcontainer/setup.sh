#!/bin/bash
set -e

# Install system deps
apt-get update && apt-get install -y cmake bear clang-format clangd