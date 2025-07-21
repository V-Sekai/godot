#ifndef IK_NODE_3D_LOCAL_H
#define IK_NODE_3D_LOCAL_H

#include "core/math/math_defs.h"
#include "core/math/transform_3d.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/math/vector3.h"

namespace kusudama {

static constexpr uint32_t IKNODE_DIRTY_LOCAL = 1;
static constexpr uint32_t IKNODE_DIRTY_GLOBAL = 2;

/**
 * Minimal, header-only local replacement for IKNode3D used for per-callsite inlining.
 *
 * This class intentionally implements only the small subset of IKNode3D behaviour
 * required by consumers (get_transform/get_global_transform, to_local/to_global,
 * set_transform, rotate_local_with_global, parent pointer handling). It's non-RefCounted
 * and lightweight so it can be embedded or constructed on the stack at call sites.
 *
 * Implementations are inline to avoid requiring a separate TU.
 */
class IKNode3DLocal {
public:
	Transform3D local_transform;
	Transform3D global_transform;
	Basis rotation;
	Vector3 scale = Vector3(1, 1, 1);
	bool disable_scale = false;
	mutable uint32_t dirty = IKNODE_DIRTY_LOCAL | IKNODE_DIRTY_GLOBAL;
	IKNode3DLocal *parent = nullptr;

	IKNode3DLocal() = default;
	~IKNode3DLocal() = default;

	inline void _update_local_transform() const {
		// local_transform.basis = rotation.scaled(scale);
		local_transform.basis = rotation.scaled(scale);
		// clear local dirty
		const_cast<IKNode3DLocal *>(this)->dirty &= ~IKNODE_DIRTY_LOCAL;
	}

	inline Transform3D get_transform() const {
		if (dirty & IKNODE_DIRTY_LOCAL) {
			_update_local_transform();
		}
		return local_transform;
	}

	inline Transform3D get_global_transform() const {
		if (dirty & IKNODE_DIRTY_GLOBAL) {
			// compute global transform
			if (parent) {
				global_transform = parent->get_global_transform() * get_transform();
			} else {
				global_transform = get_transform();
			}
			if (disable_scale) {
				global_transform.basis.orthogonalize();
			}
			const_cast<IKNode3DLocal *>(this)->dirty &= ~IKNODE_DIRTY_GLOBAL;
		}
		return global_transform;
	}

	inline void set_transform(const Transform3D &p_transform) {
		if (local_transform != p_transform) {
			local_transform = p_transform;
			// update rotation/scale from new local transform
			rotation = local_transform.basis.orthonormalized();
			// Note: keep scale as-is unless explicitly requested elsewhere
			dirty |= IKNODE_DIRTY_LOCAL;
			_propagate_transform_changed();
		}
	}

	inline void set_global_transform(const Transform3D &p_transform) {
		Transform3D xform = parent ? parent->get_global_transform().affine_inverse() * p_transform : p_transform;
		set_transform(xform);
	}

	inline Vector3 to_local(const Vector3 &p_global) const {
		return get_global_transform().affine_inverse().xform(p_global);
	}

	inline Vector3 to_global(const Vector3 &p_local) const {
		return get_global_transform().xform(p_local);
	}

	inline IKNode3DLocal *get_parent() const {
		return parent;
	}

	inline void set_parent(IKNode3DLocal *p_parent) {
		parent = p_parent;
		_propagate_transform_changed();
	}

	inline void rotate_local_with_global(const Quaternion &p_quat, bool /*p_propagate*/ = false) {
		// Convert provided global rotation into a local rotation relative to parent.
		Basis new_local_basis;
		if (parent) {
			Basis parent_global_basis = parent->get_global_transform().basis;
			Basis new_global_basis = p_quat.get_basis();
			new_local_basis = parent_global_basis.inverse() * new_global_basis;
		} else {
			new_local_basis = p_quat.get_basis();
		}
		rotation = new_local_basis.orthonormalized();
		dirty |= IKNODE_DIRTY_LOCAL | IKNODE_DIRTY_GLOBAL;
		_propagate_transform_changed();
	}

private:
	// Minimal propagate: mark children/global dirty. Consumers using IKNode3DLocal
	// that require children should implement their own child bookkeeping.
	inline void _propagate_transform_changed() {
		const_cast<IKNode3DLocal *>(this)->dirty |= IKNODE_DIRTY_GLOBAL;
	}
};

} // namespace kusudama

#endif // IK_NODE_3D_LOCAL_H
