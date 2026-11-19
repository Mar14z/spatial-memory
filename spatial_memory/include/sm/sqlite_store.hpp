#pragma once
#include <cstdint>
#include <functional>
#include <string>

struct sqlite3;
struct sqlite3_stmt;

namespace sm {

struct Entity;
struct Observation;

// Thin wrapper over SQLite (WAL mode) for the cold/persistent layer.
class SqliteStore {
public:
    SqliteStore() = default;
    ~SqliteStore();
    SqliteStore(const SqliteStore&) = delete;
    SqliteStore& operator=(const SqliteStore&) = delete;

    bool open(const std::string& path);
    void close();
    bool is_open() const { return db_ != nullptr; }

    bool begin();
    bool commit();

    bool upsert_entity(const Entity& e);
    bool append_observation(uint64_t entity_id, const Observation& o);

    void load_entities(const std::function<void(const Entity&)>& cb);
    uint64_t count_observations();

private:
    bool exec(const char* sql);

    sqlite3* db_ = nullptr;
    sqlite3_stmt* st_upsert_ = nullptr;
    sqlite3_stmt* st_obs_ = nullptr;
};

}  // namespace sm
