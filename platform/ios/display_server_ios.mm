/**************************************************************************/
/*  display_server_ios.mm                                                 */
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

#import "display_server_ios.h"

#import "device_metrics.h"

#import <UIKit/UIKit.h>
#import <sys/utsname.h>

DisplayServerIOS *DisplayServerIOS::get_singleton() {
	return (DisplayServerIOS *)DisplayServerAppleEmbedded::get_singleton();
}

DisplayServerIOS::DisplayServerIOS(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	KeyMappingIOS::initialize();

	rendering_driver = p_rendering_driver;

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		tts = [[TTS_IOS alloc] init];
	}
	native_menu = memnew(NativeMenu);

#if defined(RD_ENABLED)
	rendering_context = nullptr;
	rendering_device = nullptr;

	CALayer *layer = nullptr;

	Ref<RenderingNativeSurfaceApple> apple_surface;

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"vulkan"];
		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS Vulkan rendering layer.");
		}
		apple_surface = RenderingNativeSurfaceApple::create((__bridge void *)layer);
		rendering_context = apple_surface->create_rendering_context(rendering_driver);
	}
#endif
#ifdef METAL_ENABLED
	if (rendering_driver == "metal") {
		if (@available(iOS 14.0, *)) {
			layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"metal"];
			apple_surface = RenderingNativeSurfaceApple::create((__bridge void *)layer);
			rendering_context = apple_surface->create_rendering_context(rendering_driver);
		} else {
			OS::get_singleton()->alert("Metal is only supported on iOS 14.0 and later.");
			r_error = ERR_UNAVAILABLE;
			return;
		}
	}
#endif
	if (rendering_context) {
		if (rendering_context->initialize() != OK) {
			memdelete(rendering_context);
			rendering_context = nullptr;
#if defined(GLES3_ENABLED)
			bool fallback_to_opengl3 = GLOBAL_GET("rendering/rendering_device/fallback_to_opengl3");
			if (fallback_to_opengl3 && rendering_driver != "opengl3") {
				WARN_PRINT("Your device seem not to support MoltenVK or Metal, switching to OpenGL 3.");
				rendering_driver = "opengl3";
				OS::get_singleton()->set_current_rendering_method("gl_compatibility");
				OS::get_singleton()->set_current_rendering_driver_name(rendering_driver);
			} else
#endif
			{
				ERR_PRINT(vformat("Failed to initialize %s context", rendering_driver));
				r_error = ERR_UNAVAILABLE;
				return;
			}
		}
	}

	if (rendering_context) {
		if (rendering_context->window_create(MAIN_WINDOW_ID, apple_surface) != OK) {
			ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}

		Size2i size = Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_max_scale();
		rendering_context->window_set_size(MAIN_WINDOW_ID, size.width, size.height);
		rendering_context->window_set_vsync_mode(MAIN_WINDOW_ID, p_vsync_mode);

		rendering_device = memnew(RenderingDevice);
		if (rendering_device->initialize(rendering_context, MAIN_WINDOW_ID) != OK) {
			rendering_device = nullptr;
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}
		rendering_device->screen_create(MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		CALayer<DisplayLayer> *layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"opengl3"];

		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS OpenGLES rendering layer.");
		}

		apple_surface = RenderingNativeSurfaceApple::create((__bridge void *)layer);
		gl_manager = apple_surface->create_gl_manager();
		Ref<RenderingNativeSurface> native_surface = Ref<RenderingNativeSurface>(Object::cast_to<RenderingNativeSurface>(apple_surface.ptr()));
		[layer setupContext:gl_manager withSurface:&native_surface];

		RasterizerGLES3::make_current(false);
	}
#endif

	bool keep_screen_on = bool(GLOBAL_GET("display/window/energy_saving/keep_screen_on"));
	screen_set_keep_on(keep_screen_on);

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerIOS::~DisplayServerIOS() {
}

DisplayServer *DisplayServerIOS::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	return memnew(DisplayServerIOS(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
}

void DisplayServerIOS::register_ios_driver() {
	register_create_function("iOS", create_func, get_rendering_drivers_func);
}

String DisplayServerIOS::get_name() const {
	return "iOS";
}

int DisplayServerIOS::screen_get_dpi(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 72);

	struct utsname systemInfo;
	uname(&systemInfo);

	NSString *string = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

	NSDictionary *iOSModelToDPI = [GDTDeviceMetrics dpiList];

	for (NSArray *keyArray in iOSModelToDPI) {
		if ([keyArray containsObject:string]) {
			NSNumber *value = iOSModelToDPI[keyArray];
			return [value intValue];
		}
	}

	// If device wasn't found in dictionary
	// make a best guess from device metrics.
	CGFloat scale = [UIScreen mainScreen].scale;

	UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;

	switch (idiom) {
		case UIUserInterfaceIdiomPad:
			return scale == 2 ? 264 : 132;
		case UIUserInterfaceIdiomPhone: {
			if (scale == 3) {
				CGFloat nativeScale = [UIScreen mainScreen].nativeScale;
				return nativeScale == 3 ? 458 : 401;
			}

			return 326;
		}
		default:
			return 72;
	}
}

float DisplayServerIOS::screen_get_refresh_rate(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

	float fps = [UIScreen mainScreen].maximumFramesPerSecond;
	if ([NSProcessInfo processInfo].lowPowerModeEnabled) {
		fps = 60;
	}
	return fps;
}

float DisplayServerIOS::screen_get_scale(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 1.0f);

	return [UIScreen mainScreen].scale;
}

Vector<DisplayServer::WindowID> DisplayServerIOS::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerIOS::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

int64_t DisplayServerIOS::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, 0);
	switch (p_handle_type) {
		case DISPLAY_HANDLE: {
			return 0; // Not supported.
		}
		case WINDOW_HANDLE: {
			return (int64_t)AppDelegate.viewController;
		}
		case WINDOW_VIEW: {
			return (int64_t)AppDelegate.viewController.godotView;
		}
#if defined(GLES3_ENABLED)
		case OPENGL_FBO: {
			if (gl_manager) {
				return (int64_t)gl_manager->window_get_render_target(DisplayServer::MAIN_WINDOW_ID);
			}
			return 0;
		}
#endif
		default: {
			return 0;
		}
	}
}

void DisplayServerIOS::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerIOS::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerIOS::window_set_title(const String &p_title, WindowID p_window) {
	// Probably not supported for iOS
}

int DisplayServerIOS::window_get_current_screen(WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerIOS::window_set_current_screen(int p_screen, WindowID p_window) {
	// Probably not supported for iOS
}

Point2i DisplayServerIOS::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerIOS::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerIOS::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerIOS::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Probably not supported for iOS
}

void DisplayServerIOS::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIOS::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIOS::window_set_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_size(WindowID p_window) const {
	CGRect screenBounds = [UIScreen mainScreen].bounds;
	return Size2i(screenBounds.size.width, screenBounds.size.height) * screen_get_max_scale();
}

Size2i DisplayServerIOS::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerIOS::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Probably not supported for iOS
}

DisplayServer::WindowMode DisplayServerIOS::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerIOS::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerIOS::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerIOS::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerIOS::window_request_attention(WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerIOS::window_move_to_foreground(WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerIOS::window_is_focused(WindowID p_window) const {
	return true;
}

float DisplayServerIOS::screen_get_max_scale() const {
	return screen_get_scale(SCREEN_OF_MAIN_WINDOW);
}

void DisplayServerIOS::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	screen_orientation = p_orientation;
	if (@available(iOS 16.0, *)) {
		[AppDelegate.viewController setNeedsUpdateOfSupportedInterfaceOrientations];
	} else {
		[UIViewController attemptRotationToDeviceOrientation];
	}
}

DisplayServer::ScreenOrientation DisplayServerIOS::screen_get_orientation(int p_screen) const {
	return screen_orientation;
}

bool DisplayServerIOS::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerIOS::can_any_window_draw() const {
	return true;
}

bool DisplayServerIOS::is_touchscreen_available() const {
	return true;
}

_FORCE_INLINE_ int _convert_utf32_offset_to_utf16(const String &p_existing_text, int p_pos) {
	int limit = p_pos;
	for (int i = 0; i < MIN(p_existing_text.length(), p_pos); i++) {
		if (p_existing_text[i] > 0xffff) {
			limit++;
		}
	}
	return limit;
}

void DisplayServerIOS::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
	AppDelegate.viewController.keyboardView.textContentType = nil;
	switch (p_type) {
		case KEYBOARD_TYPE_DEFAULT: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_MULTILINE: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_NUMBER: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeNumberPad;
		} break;
		case KEYBOARD_TYPE_NUMBER_DECIMAL: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDecimalPad;
		} break;
		case KEYBOARD_TYPE_PHONE: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypePhonePad;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeTelephoneNumber;
		} break;
		case KEYBOARD_TYPE_EMAIL_ADDRESS: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeEmailAddress;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeEmailAddress;
		} break;
		case KEYBOARD_TYPE_PASSWORD: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypePassword;
		} break;
		case KEYBOARD_TYPE_URL: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeWebSearch;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeURL;
		} break;
	}

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
							   cursorStart:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_start)
								 cursorEnd:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_end)];
}

bool DisplayServerIOS::is_keyboard_active() const {
	return [AppDelegate.viewController.keyboardView isFirstResponder];
}

void DisplayServerIOS::virtual_keyboard_hide() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

void DisplayServerIOS::virtual_keyboard_set_height(int height) {
	virtual_keyboard_height = height * screen_get_max_scale();
}

int DisplayServerIOS::virtual_keyboard_get_height() const {
	return virtual_keyboard_height;
}

bool DisplayServerIOS::has_hardware_keyboard() const {
	if (@available(iOS 14.0, *)) {
		return [GCKeyboard coalescedKeyboard];
	} else {
		return false;
	}
}

void DisplayServerIOS::clipboard_set(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8()];
}

String DisplayServerIOS::clipboard_get() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}

void DisplayServerIOS::screen_set_keep_on(bool p_enable) {
	[UIApplication sharedApplication].idleTimerDisabled = p_enable;
}

bool DisplayServerIOS::screen_is_kept_on() const {
	return [UIApplication sharedApplication].idleTimerDisabled;
}

void DisplayServerIOS::resize_window(CGSize viewSize) {
	Size2i size = Size2i(viewSize.width, viewSize.height) * screen_get_max_scale();

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(MAIN_WINDOW_ID, size.x, size.y);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callback, resize_rect);
}

void DisplayServerIOS::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerIOS::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}
