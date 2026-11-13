# Wearable meeting agent pipeline

A design for a wearable that records conversations in the physical world and turns them into summaries that say who said and committed to what, with speakers recognized across conversations.

- [System design](docs/system-design.md): architecture from microphone to summary, data model, interfaces, failure modes, scale and cost, the seven scenarios, and privacy.
- [Deep dive: speaker identity service](docs/system-design.md#4-deep-dive-speaker-identity-service): how voices are matched, learned, corrected and deleted, with thresholds taken from the build.
- [Build: voiceprint matching on far-field audio](build/README.md): an off-the-shelf speaker embedding model evaluated on AMI meeting audio, with strangers who were never enrolled and with nearby talkers mixed in.
