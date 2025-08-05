/**************************************************************************/
/*  clap_helpers_instantiation.cpp                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <clap/helpers/checking-level.hh>
#include <clap/helpers/host.hxx>
#include <clap/helpers/misbehaviour-handler.hh>
#include <clap/helpers/plugin-proxy.hxx>
#include <clap/ext/gui.h>
#include <clap/ext/params.h>
#include <clap/ext/audio-ports.h>
#include <clap/ext/state.h>
#include <clap/ext/latency.h>
#include <clap/ext/note-ports.h>
#include <clap/ext/posix-fd-support.h>
#include <clap/ext/preset-load.h>
#include <clap/ext/remote-controls.h>
#include <clap/ext/render.h>
#include <clap/ext/tail.h>
#include <clap/ext/thread-pool.h>
#include <clap/ext/timer-support.h>

// Explicit template instantiation for the specific template parameters used in the CLAP module
// Using Terminate misbehaviour handler and Maximal checking level
template class clap::helpers::Host<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>;
template class clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>;

// Explicit instantiation of the getExtension template method for all extension types used
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_gui>(const clap_plugin_gui*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_params>(const clap_plugin_params*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_audio_ports>(const clap_plugin_audio_ports*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_state>(const clap_plugin_state*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_latency>(const clap_plugin_latency*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_note_ports>(const clap_plugin_note_ports*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_posix_fd_support>(const clap_plugin_posix_fd_support*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_preset_load>(const clap_plugin_preset_load*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_remote_controls>(const clap_plugin_remote_controls*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_render>(const clap_plugin_render*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_tail>(const clap_plugin_tail*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_thread_pool>(const clap_plugin_thread_pool*&, const char*) const noexcept;
template void clap::helpers::PluginProxy<clap::helpers::MisbehaviourHandler::Terminate, clap::helpers::CheckingLevel::Maximal>::getExtension<clap_plugin_timer_support>(const clap_plugin_timer_support*&, const char*) const noexcept;
