# Audit: /etc/spar/node_env and its consumers (2026-09-21)

Four separate failures in two days turned out to be one bug repeated. This is the
audit I should have done before adding a second hardware track, not after.

## The shape of the bug

`bootstrap_node.sh` sources `env.sh` at the top. `env.sh` **exports** hand-run
defaults for `BUCKET`, `MASTER_ADDR`, `MASTER_PORT`, `NODE0`, `NODE1`, `PROJECT`,
`SESSION` and `ZONE`. Bootstrap then resolves each with

```sh
X="${X:-$(meta spar-x)}"
```

which **can never reach the metadata**, because `X` is already set. Bootstrap
writes the result into `/etc/spar/node_env`, which every other script sources and
treats as authoritative. So `env.sh`'s defaults — `spar-gpu-0`, `spar-gpu-1`,
`us-central1-a` — were being presented to the whole system as the node's real
identity.

None of it shows up on a pair actually named `spar-gpu-0/1` in `us-central1-a`,
which is every run this project did before 2026-09-19. It surfaces the moment a
pair is named anything else — i.e. on every pair the multi-zone fan-out creates.

## What was wrong, measured on the live A100-80GB pair in asia-southeast1-c

| variable | node_env said | truth | consumer that broke |
|---|---|---|---|
| `MASTER_ADDR` | `spar-gpu-0` | `spar-gpua-0` | Ray formed no cluster; every 2-node cell `ray-cluster-incomplete` |
| `NODE0` / `NODE1` | `spar-gpu-0` / `-1` | `spar-gpua-0` / `-1` | `cell_barrier` peer, `launch_queue` shutdown rendezvous, `verify_pair` |
| `ZONE` | `us-central1-a` | `asia-southeast1-c` | `verify_pair`'s `describe`/`ssh --zone` → reports the peer "not provisioned" |
| `SESSION` | correct by coincidence | — | GCS prefix: wrong value files telemetry under the wrong run |
| `MASTER_PORT` | correct by coincidence | — | latent |
| `BUCKET` | correct by coincidence | — | latent, and the worst one: telemetry would sync to a bucket nobody reads |

`PROJECT` is right only because there is one project.

The three "correct by coincidence" rows are the reason this went unnoticed for so
long: the mechanism was broken for all six variables, and three of them happened
to agree with the truth.

## Consumers

Everything that sources `node_env`:

| script | variables used |
|---|---|
| `bootstrap_node.sh` | writes it |
| `env.sh` | re-sources it |
| `launch_queue.sh` | `BUCKET SESSION NODE0 NODE1 MASTER_ADDR MASTER_PORT` |
| `run_stage_and_stop.sh` | `BUCKET SESSION` |
| `sync_data.sh` | `BUCKET SESSION` |
| `multinode_queue.sh` | `BUCKET SESSION NODE0 NODE1 MASTER_ADDR MASTER_PORT` |
| `verify_pair.sh` | `NODE0 NODE1 ZONE PROJECT` |

Two of them derived the peer with the same unsafe idiom:

```sh
[ "$NODE_NAME" = "$NODE0" ] && PEER="$NODE1" || PEER="$NODE0"
```

If `NODE_NAME` matches **neither** name this silently yields `PEER=$NODE0` — a
host that does not exist. In `cell_barrier` that meant the nodes never aligned;
in `launch_queue`'s `STOP_WHEN_DONE` it meant both nodes would idle until
`PEER_WAIT` (2400 s) expired instead of powering off.

## Fixes

1. **Metadata is forced, not defaulted.** A `_force VAR spar-key` helper
   overwrites `SESSION`, `MASTER_PORT` and `BUCKET` when the key exists.
   `MASTER_ADDR` was fixed the same way in 6cfa49f.
2. **Zone and project come from the metadata server itself**
   (`instance/zone`, `project/project-id`) rather than a `spar-*` key, so they
   are right without any provisioner cooperation.
3. **`NODE0`/`NODE1` come from metadata.** A MIG template is shared by both
   nodes, so a per-node `spar-peer-name` cannot exist on that path; the MIG
   provisioner now ships `spar-node0-name` and `spar-node1-name`, and
   `NODE_INDEX` says which is which. `spar-peer-name` is still honoured for the
   per-instance provisioners.
4. **Peer derivation refuses to guess.** Both `cell_barrier` and
   `launch_queue` now prefer an explicit `PEER`, fall back to `NODE0`/`NODE1`
   **only when `NODE_NAME` is actually one of them**, and otherwise warn loudly
   instead of silently selecting a wrong host.

Verified offline against all five provisioning shapes — MIG node0/node1,
per-instance node0/node1, and legacy `spar-gpu-0/1` with no metadata — checking
both the resolved `NODE0`/`NODE1`/`ZONE` and the peer each node derives.

## Rule for anything added later

Any value a node needs about **itself or its pair** must come from instance
metadata or the metadata server. `env.sh` is for driving provisioning from a
laptop; its values are guesses about a node that does not exist yet, and they
must never win once the node does. If a consumer cannot resolve an identity, it
must say so — the silent fallbacks are what made four bugs take two days instead
of one audit.
