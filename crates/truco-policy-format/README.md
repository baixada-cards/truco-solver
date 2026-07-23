# truco-policy-format

`truco-policy-format` is the versioned interchange contract between a Truco
policy producer and a runtime bot. It contains no CFR trainer, checkpoint
loader, experiment harness, or hosted-game code.

The crate owns:

- the abstract card and action vocabulary;
- deterministic information-set construction and keys;
- the `TPB1` mmap-friendly policy file codec;
- the `truco-policy-bot/v1` bundle manifest and its JSON Schema.

That ownership makes the repository boundary intentional: `truco-solver`
produces policy bundles, while `truco-bots` consumes them by depending only on
this crate and `truco-engine`.

## TPB1 binary layout

All integers are little-endian.

| Field | Size | Meaning |
| --- | ---: | --- |
| Magic | 4 bytes | ASCII `TPB1` |
| Version | 2 bytes | `1` |
| Reserved | 2 bytes | `0` |
| Record count | 8 bytes | Unsigned 64-bit count |
| Records | `count × 24` bytes | Fixed-width entries |

Each record contains an 8-byte information-set key, eight one-byte abstract
action codes, and eight one-byte probabilities. Unused action slots contain
`0xFF`; the corresponding probability slots are zero. Probabilities are
quantized to `0..=255` and renormalized when read.

Records are ordered lexicographically by the key's encoded little-endian bytes.
This is the historical v1 ordering and is part of the compatibility contract;
it is not ordinary numeric `u64` ordering.

## Bundle manifest

A bundle is a directory containing `manifest.json` and one or more `.tpb`
files. The manifest discriminator is `truco-policy-bot/v1`. Filenames must be
single, bundle-local `.tpb` names—absolute paths and directory traversal are
rejected. See
[`schema/policy-manifest-v1.schema.json`](schema/policy-manifest-v1.schema.json).

## Compatibility policy

- Existing v1 bytes and deterministic keys must remain readable forever.
- Compatible readers may add validation without changing accepted valid bytes.
- Any change to key derivation, action codes, record layout, ordering, or
  probability encoding requires a new format version and migration fixtures.
- Releases should include golden key and binary fixtures plus checksums.

The format describes redistributable policy metadata only. Large trained
policies and private research artifacts are deployed from private object
storage and are not committed to public Git history.
