#ifndef BIMDF_WRAPPER_H
#define BIMDF_WRAPPER_H

#include "core/object/ref_counted.h"
#include <cstdint>
#include <libsatsuma/Problems/BiMDF.hh>

#include <map>

class BIMDFSolver : public RefCounted {
	GDCLASS(BIMDFSolver, RefCounted);

    Satsuma::BiMDF bimdf;
    std::map<String, Satsuma::BiMDF::Edge> edges;

protected:
    static void _bind_methods();

public:
    // int add_node();
    // void add_edge(const String &name, int u, lemon::ListGraphBase::Node v, bool u_head, bool v_head, float target, float weight, int64_t lower);
    Dictionary solve();

    BIMDFSolver();
};

#endif // BIMDF_WRAPPER_H
