---
title: Arietta Voice Capabilities
tags: assistant, capabilities, tools, knowledge, setup
aliases: what can you help with
---

## Core Behavior

Bridget is a local voice assistant that wakes when it hears the configured wake word and can keep a short conversation open for follow-up questions.

## Knowledge

The assistant can answer questions from markdown files stored in the local knowledge directory. Users should add their own articles for project, household, company, or device-specific facts.

## Tools

The assistant supports two tool styles. Deterministic tools are used for exact, pre-routed actions like checking the local time. Model-requested tools are used when the language model should decide whether a follow-up lookup or action is needed.

## Customization

Users can customize the assistant by editing the SOUL prompt, changing the wake word, adding knowledge articles, and creating new tools for lookups or actions such as home automation.
