ARG NODE_VER=23.5.0
ARG BASE_IMAGE=node:${NODE_VER}
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

# Install pnpm globally and install necessary build tools
RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    make \
    g++ \
    nano \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PNPM_VER=9.15.2
RUN npm install -g pnpm@${PNPM_VER}

# Set Python 3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV DEBIAN_FRONTEND=dialog
