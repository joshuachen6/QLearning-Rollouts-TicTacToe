#include "Table.h"

Table::Table(std::string path) {
    int code = sqlite3_open(path.c_str(), &db);
    if (code != SQLITE_OK) {
        std::cerr << "Failed to Open Database File" << std::endl;
    }
    const char* cmd = "CREATE TABLE IF NOT EXISTS q_values (state BLOB PRIMARY KEY, value REAL, count INTEGER DEFAULT 0);";
    sqlite3_exec(db, cmd, NULL, NULL, NULL);
}

Table& Table::updateQ(torch::Tensor& state, double value) {
    std::lock_guard lock(mutex);
    sqlite3_stmt* stmt;
    std::string cmd = 
        "INSERT INTO q_values (state, value) "
        "VALUES (?, ?) "
        "ON CONFLICT(state) DO UPDATE SET "
        "value = (value * count + excluded.value) / (count + 1), "
        "count = count + 1;";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);

    std::string data = serialize(state);
    sqlite3_bind_blob(stmt, 1, data.data(), data.size(), SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, value);
    int code = sqlite3_step(stmt);
    if (code != SQLITE_DONE) {
        std::cerr << sqlite3_errmsg(db) << std::endl;;
    }

    sqlite3_finalize(stmt);
    return *this;
}

Table& Table::updateQ(std::vector<std::vector<torch::Tensor>>& states, std::vector<std::vector<double>>& values) {
    std::lock_guard lock(mutex);
    sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    sqlite3_stmt* stmt;
    std::string cmd = 
        "INSERT INTO q_values (state, value, count) "
        "VALUES (?, ?, 1) "
        "ON CONFLICT(state) DO UPDATE SET "
        "value = value + 0.3 * (excluded.value - value), "
        "count = count + 1;";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);

    for (int i = 0; i < states.size(); i++) {
        for (int j = 0; j < states[i].size(); j++) {
            torch::Tensor state = states[i][j];
            double val = values[i][j];

            // Add 8 symmetries (4 rotations * 2 flips)
            for (int r = 0; r < 4; r++) {
                // Rotation
                torch::Tensor rotated = state.clone();
                if (r > 0) {
                    rotated.index_put_({0}, torch::rot90(state.index({0}), r, {0, 1}));
                    rotated.index_put_({1}, torch::rot90(state.index({1}), r, {0, 1}));
                }

                auto addSym = [&](torch::Tensor s) {
                    std::string data = serialize(s);
                    sqlite3_bind_blob(stmt, 1, data.data(), data.size(), SQLITE_STATIC);
                    sqlite3_bind_double(stmt, 2, val);
                    sqlite3_step(stmt);
                    sqlite3_reset(stmt);
                    sqlite3_clear_bindings(stmt);
                };

                addSym(rotated);

                // Flip
                torch::Tensor flipped = rotated.clone();
                flipped.index_put_({0}, torch::flip(rotated.index({0}), {0}));
                flipped.index_put_({1}, torch::flip(rotated.index({1}), {0}));
                addSym(flipped);
            }
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_exec(db, "END TRANSACTION;", NULL, NULL, NULL);
    return *this;
}

double Table::getQ(torch::Tensor& state) {
    std::lock_guard lock(mutex);
    sqlite3_stmt* stmt;
    std::string cmd = "SELECT value FROM q_values WHERE state = ?";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
    std::string data = serialize(state);
    sqlite3_bind_blob(stmt, 1, data.data(), data.size(), SQLITE_STATIC);

    double value;
    int code;
    code = sqlite3_step(stmt);
    if (code == SQLITE_ROW) {
        value = sqlite3_column_double(stmt, 0);
    } else {
        std::cerr << "Failed to retrieve Q-value" << std::endl;
    }

    sqlite3_finalize(stmt);
    return value;
}

std::pair<torch::Tensor, torch::Tensor> Table::getDataset(int size) {
    std::lock_guard lock(mutex);
    sqlite3_stmt* stmt;
    std::string cmd = "SELECT state, value FROM q_values ORDER BY RANDOM() LIMIT ?";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, size);

    torch::Tensor y = torch::zeros({size});
    std::vector<torch::Tensor> xBuffer;

    int code;
    for (int i = 0;;i++) {
        code = sqlite3_step(stmt);
        if (code == SQLITE_ROW) {
            const void* tensorData = sqlite3_column_blob(stmt, 0);
            int dataSize = sqlite3_column_bytes(stmt, 0);
            xBuffer.push_back(deserialize(tensorData, dataSize));
            y[i] = sqlite3_column_double(stmt, 1);
        } else if (code == SQLITE_DONE) {
            break;
        }
    }

    y = y.slice(0, 0, xBuffer.size());
    
    torch::Tensor x = torch::stack(xBuffer);
    sqlite3_finalize(stmt);
    return std::pair<torch::Tensor, torch::Tensor>(x, y);
}

Table::~Table() {
    sqlite3_close(db);
}


std::string serialize(torch::Tensor tensor) {
    std::ostringstream stream;
    torch::save(tensor, stream);
    return stream.str();
}

torch::Tensor deserialize(const void* data, int size) {
    std::string str((char*) data, size);
    std::istringstream stream(str);
    torch::Tensor tensor;
    torch::load(tensor, stream);
    return tensor;
}