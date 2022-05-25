# Alignment as a Multi-agent Intrinsic Reward

This repository contains code to train and evaluate multiple agents with and without 
the alignment intrinsic reward in both the multi-agent particle and Google research 
football environments.

[[Paper]]()

## Abstract

Modern multi-agent reinforcement learning frameworks rely on centralized training and reward shaping to perform well.
However, centralized training and dense rewards are not readily available in the real world. 
Current multi-agent algorithms struggle to learn in the alternative setup of decentralized training or sparse rewards.
To address these issues, we propose a self-supervised intrinsic reward called alignment inspired by the self-organization principle in Zoology.
Similar to how animals collaborate in a decentralized manner with those in their vicinity, agents trained with alignment learn behaviors that match their neighbors' expectations.
This allows the agents to learn collaborative behaviors without any external reward or centralized training.
We demonstrate the efficacy of our approach across 6 tasks in the multi-agent particle and the complex Google Research football environments, comparing alignment to sparse and curiosity-based intrinsic rewards.
When the number of agents increases, alignment scales well in all multi-agent tasks except for one where agents have different capabilities.
We show that agent coordination improves through alignment because agents learn to divide tasks amongst themselves, break coordination symmetries, and confuse adversaries.
These results identify tasks where alignment is a more useful strategy than curiosity-driven exploration for multi-agent coordination, enabling agents to do zero-shot coordination.

## Alignment illustraion

| ![Alignment](https://github.com/StanfordVL/alignment/raw/main/docs/alignment.png) |
|:--:|

## Contents
- [Installation](#installation)
- [Multi-agent particle](#map)
- [Google research football](#gfootball)

## Installation

## Multi-agent particle

### Training

### Evaluation

## Google research football


