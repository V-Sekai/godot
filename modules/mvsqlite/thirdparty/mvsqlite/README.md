# mvSQLite

mvSQLite is a distributed, MVCC SQLite running on [FoundationDB](https://github.com/apple/foundationdb), modified by V-Sekai (https://github.com/V-Sekai). It offers full feature-set from SQLite, time travel, lock-free scalable reads and writes, and more. You can use `LD_PRELOAD` or a patched `libsqlite3.so` to integrate mvSQLite into your existing apps.

## Quick Links

- [Documentation](https://github.com/losfair/mvsqlite/wiki/)
- [Releases](https://github.com/losfair/mvsqlite/releases)
- [Quick Reference](https://github.com/V-Sekai/mvsqlite/wiki/Quick-Reference)

## Getting Started

1. Install FoundationDB:

```bash
wget https://github.com/apple/foundationdb/releases/download/7.1.15/foundationdb-clients_7.1.15-1_amd64.deb
sudo dpkg -i foundationdb-clients_7.1.15-1_amd64.deb
wget https://github.com/apple/foundationdb/releases/download/7.1.15/foundationdb-server_7.1.15-1_amd64.deb
sudo dpkg -i foundationdb-server_7.1.15-1_amd64.deb
```

2. Build, run `mvstore`, create a namespace with the admin API, build `libsqlite3` and the `sqlite3` CLI, set environment variables, and run the shell. Detailed instructions can be found in the [wiki](https://github.com/V-Sekai/mvsqlite/wiki).


```bash
cargo build --release -p mvstore
RUST_LOG=info ./mvstore \
  --data-plane 127.0.0.1:7000 \
  --admin-api 127.0.0.1:7001 \
  --metadata-prefix mvstore \
  --raw-data-prefix m
```

Create a namespace with the admin API:

```bash
curl http://localhost:7001/api/create_namespace -i -d '{"key":"test"}'
```

3. To run `sqlite3`. Build `libsqlite3` and the `sqlite3` CLI: (note that a custom build is only needed here because the `sqlite3` binary shipped on most systems are statically linked to `libsqlite3` and `LD_PRELOAD` don't work)

```bash
cargo build --release -p mvsqlite
cd mvsqlite-sqlite3
make build-patched-sqlite3
./sqlite3
```

## Contributing

mvSQLite can be built with the standard Rust toolchain. More details are available in the [wiki](https://github.com/V-Sekai/mvsqlite/wiki).
