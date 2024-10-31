#include "dictionary.hpp"

#include "syscalls.h"

EXTERN_SYSCALL(void, sys_vcreate, Variant *, int, int, const void *);
MAKE_SYSCALL(ECALL_DICTIONARY_OPS, int, sys_dict_ops, Dictionary_Op, unsigned, const Variant *, Variant *);
MAKE_SYSCALL(ECALL_DICTIONARY_OPS, int, sys_dict_ops2, Dictionary_Op, unsigned, const Variant *, Variant *, const Variant *);
EXTERN_SYSCALL(unsigned, sys_vassign, unsigned, unsigned);

Dictionary &Dictionary::operator=(const Dictionary &other) {
	this->m_idx = sys_vassign(this->m_idx, other.m_idx);
	return *this;
}

void Dictionary::clear() {
	(void)sys_dict_ops(Dictionary_Op::CLEAR, m_idx, nullptr, nullptr);
}

void Dictionary::erase(const Variant &key) {
	(void)sys_dict_ops(Dictionary_Op::ERASE, m_idx, &key, nullptr);
}

bool Dictionary::has(const Variant &key) const {
	return sys_dict_ops(Dictionary_Op::HAS, m_idx, &key, nullptr);
}

int Dictionary::size() const {
	return sys_dict_ops(Dictionary_Op::GET_SIZE, m_idx, nullptr, nullptr);
}

Variant Dictionary::get(const Variant &key) const {
	Variant v;
	(void)sys_dict_ops(Dictionary_Op::GET, m_idx, &key, &v);
	return v;
}
void Dictionary::set(const Variant &key, const Variant &value) {
	(void)sys_dict_ops(Dictionary_Op::SET, m_idx, &key, (Variant *)&value);
}
Variant Dictionary::get_or_add(const Variant &key, const Variant &default_value) {
	Variant v;
	(void)sys_dict_ops2(Dictionary_Op::GET_OR_ADD, m_idx, &key, &v, &default_value);
	return v;
}

void Dictionary::merge(const Dictionary &other) {
	Variant v(other);
	(void)sys_dict_ops(Dictionary_Op::MERGE, m_idx, &v, nullptr);
}

Dictionary Dictionary::Create() {
	Variant v;
	sys_vcreate(&v, Variant::DICTIONARY, 0, nullptr);
	Dictionary d;
	d.m_idx = v.get_internal_index();
	return d;
}
