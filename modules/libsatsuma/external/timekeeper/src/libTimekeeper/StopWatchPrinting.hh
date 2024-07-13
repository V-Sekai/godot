//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once

#include <libTimekeeper/StopWatch.hh>

#include <string_view>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cstddef>
#include <array>

namespace Timekeeper {


inline size_t utf8_len(std::string const&s) {
  return count_if(s.begin(), s.end(), [](char c)->bool { return (c & 0xC0) != 0x80; });
}

inline std::string strrep(const std::string_view s, int n) {
    if (n <= 0)
        return "";
    std::string indent;
    indent.reserve(s.size() * n);
    for (int i = 0; i < n; ++i) {
        indent.append(s);
    }
    return indent;
}

template<size_t NCols>
class PrintedTable {
public:
  enum class Align { Left, Right };
  using Row = std::array<std::string, NCols>;
  using ColAlignment = std::array<Align, NCols>;

  constexpr PrintedTable(ColAlignment _align)
    : align_(std::move(_align))
  {
    std::fill(widths_.begin(), widths_.end(), 0);
  }

  template<typename...T>
  void add_row(Row row) {
    rows_.emplace_back(std::move(row));
    for(size_t i = 0; i < NCols; ++i) {
      widths_[i] = std::max(widths_[i], utf8_len(rows_.back()[i]));
    }
  }
  void print(std::ostream &os) const {
    std::ios oldState(nullptr);
    oldState.copyfmt(os);

    for (const auto &row: rows_) {
      for(size_t i = 0; i < NCols; ++i) {
        // std::setw on an ostream(!) performs byte-wise padding,
        // so we do it ourselves with the correct sizes.
        auto pad = strrep(" ", static_cast<int>(widths_[i] - utf8_len(row[i])));
        if (align_[i] == Align::Right) {
          os << pad;
        }
        os << row[i];
        if (align_[i] == Align::Left) {
          os << pad;
        }

        // inter-column spacing:
        if (i + 1 < NCols) {
          os << std::setw(0) << "  ";
        }
      }
      os << std::endl;
    }
    os.copyfmt(oldState);
  }

private:
    std::vector<Row> rows_;
    std::array<size_t, NCols> widths_;
    ColAlignment align_;
};

template<size_t NCols>
inline std::ostream &operator<<(std::ostream &os, const PrintedTable<NCols> &table)
{
  table.print(os);
  return os;
}

template<typename ChronoDuration>
inline std::string format_ms(ChronoDuration dur) {
  return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()).append(" ms");
}


using PT = PrintedTable<5>;

#ifdef _WIN32
inline const std::string symbol_branch ="+ ";
inline const std::string symbol_vert = "| ";
inline const std::string symbol_lastbranch = "\ ";
#else
inline const std::string symbol_branch ="├ ";
inline const std::string symbol_vert = "│ ";
inline const std::string symbol_lastbranch = "╰ ";
#endif

inline void add_to_table(PT &table, const HierarchicalStopWatchResult &wr, size_t depth=0)
{
    std::string tree = strrep(symbol_vert, depth > 1 ? depth - 1 : 0);
    std::string tree_name = tree;
    if (depth > 0) {
        tree_name.append(symbol_branch);
    }
    tree_name.append(wr.name);
    std::string dur_wall, dur_user, dur_sys, count;
    dur_wall = format_ms(wr.duration.wallclock);
    dur_user = format_ms(wr.duration.user);
    dur_sys  = format_ms(wr.duration.system);
    count = std::to_string(wr.count);
    if (wr.is_group()) {
      table.add_row({tree_name, "", "", "", ""});
    } else {
      table.add_row({tree_name, dur_wall, dur_user, dur_sys, count});
    }

    for (auto const &child: wr.children) {
        add_to_table(table, child, depth+1);
    }
    if (!wr.children.empty())
    {
      std::string tree_total = tree;
      if (depth > 0) {
        tree_total.append(symbol_vert);
      }
      tree_total.append(symbol_lastbranch);
      if (!wr.is_group()) {
        tree_total.append("(unaccounted)");
        auto unaccounted = wr.unaccounted();
        dur_wall = format_ms(unaccounted.wallclock);
        dur_user = format_ms(unaccounted.user);
        dur_sys  = format_ms(unaccounted.system);
        table.add_row({tree_total, dur_wall, dur_user, dur_sys, ""});
      } else {
        tree_total.append("(end group)");
        table.add_row({tree_total, "", "", "", ""});
      }
    }
}
inline std::ostream &operator<<(std::ostream &os, const HierarchicalStopWatchResult &sw_result)
{
  using Align = typename PT::Align;
  auto table = PT{{Align::Left, Align::Right, Align::Right, Align::Right, Align::Right}};
  table.add_row({
      std::string(""),
      std::string("Wall"),
      std::string("User"),
      std::string("System"),
      std::string("count")
      });
  add_to_table(table, sw_result);
  os << table;
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const HierarchicalStopWatch &sw)
{
    return os << HierarchicalStopWatchResult(sw);
}

} // namespace Timekeeper
