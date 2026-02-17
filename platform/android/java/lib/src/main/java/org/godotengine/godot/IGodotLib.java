package org.godotengine.godot;

import android.app.Activity;
import android.content.res.AssetManager;
import android.hardware.SensorEvent;
import android.view.Surface;

import org.godotengine.godot.gl.GodotRenderer;
import org.godotengine.godot.io.directory.DirectoryAccessHandler;
import org.godotengine.godot.io.file.FileAccessHandler;
import org.godotengine.godot.tts.GodotTTS;
import org.godotengine.godot.utils.GodotNetUtils;
import org.godotengine.godot.variant.Callable;

import javax.microedition.khronos.opengles.GL10;

public interface IGodotLib {

	/**
	 * Invoked on the main thread to initialize Godot native layer.
	 */
	public boolean initialize(
		Godot p_instance,
		AssetManager p_asset_manager,
		GodotIO godotIO,
		GodotNetUtils netUtils,
		DirectoryAccessHandler directoryAccessHandler,
		FileAccessHandler fileAccessHandler,
		boolean use_apk_expansion);

	/**
	 * Invoked on the main thread to clean up Godot native layer.
	 * @see androidx.fragment.app.Fragment#onDestroy()
	 */
	public void ondestroy();

	/**
	 * Invoked on the GL thread to complete setup for the Godot native layer logic.
	 * @param p_cmdline Command line arguments used to configure Godot native layer components.
	 */
	public boolean setup(String[] p_cmdline, GodotTTS tts);

	/**
	 * Invoked on the GL thread when the underlying Android surface has changed size.
	 * @param p_surface
	 * @param p_width
	 * @param p_height
	 * @see org.godotengine.godot.gl.GLSurfaceView.Renderer#onSurfaceChanged(GL10, int, int)
	 */
	public void resize(Surface p_surface, int p_width, int p_height);

	/**
	 * Invoked on the render thread when the underlying Android surface is created or recreated.
	 * @param p_surface
	 */
	public void newcontext(Surface p_surface);

	/**
	 * Forward {@link Activity#onBackPressed()} event.
	 */
	public void back();

	/**
	 * Invoked on the GL thread to draw the current frame.
	 * @see org.godotengine.godot.gl.GLSurfaceView.Renderer#onDrawFrame(GL10)
	 */
	public boolean step();

	/**
	 * TTS callback.
	 */
	public void ttsCallback(int event, int id, int pos);

	/**
	 * Forward touch events.
	 */
	public void dispatchTouchEvent(int event, int pointer, int pointerCount, float[] positions, boolean doubleTap);

	/**
	 * Dispatch mouse events
	 */
	public void dispatchMouseEvent(int event, int buttonMask, float x, float y, float deltaX, float deltaY, boolean doubleClick, boolean sourceMouseRelative, float pressure, float tiltX, float tiltY);

	public void magnify(float x, float y, float factor);

	public void pan(float x, float y, float deltaX, float deltaY);

	/**
	 * Forward accelerometer sensor events.
	 * @see android.hardware.SensorEventListener#onSensorChanged(SensorEvent)
	 */
	public void accelerometer(float x, float y, float z);

	/**
	 * Forward gravity sensor events.
	 * @see android.hardware.SensorEventListener#onSensorChanged(SensorEvent)
	 */
	public void gravity(float x, float y, float z);

	/**
	 * Forward magnetometer sensor events.
	 * @see android.hardware.SensorEventListener#onSensorChanged(SensorEvent)
	 */
	public void magnetometer(float x, float y, float z);

	/**
	 * Forward gyroscope sensor events.
	 * @see android.hardware.SensorEventListener#onSensorChanged(SensorEvent)
	 */
	public void gyroscope(float x, float y, float z);

	/**
	 * Forward regular key events.
	 */
	public void key(int p_physical_keycode, int p_unicode, int p_key_label, boolean p_pressed, boolean p_echo);

	/**
	 * Forward game device's key events.
	 */
	public void joybutton(int p_device, int p_but, boolean p_pressed);

	/**
	 * Forward joystick devices axis motion events.
	 */
	public void joyaxis(int p_device, int p_axis, float p_value);

	/**
	 * Forward joystick devices hat motion events.
	 */
	public void joyhat(int p_device, int p_hat_x, int p_hat_y);

	/**
	 * Fires when a joystick device is added or removed.
	 */
	public void joyconnectionchanged(int p_device, boolean p_connected, String p_name);

	/**
	 * Invoked when the Android app resumes.
	 * @see androidx.fragment.app.Fragment#onResume()
	 */
	public void focusin();

	/**
	 * Invoked when the Android app pauses.
	 * @see androidx.fragment.app.Fragment#onPause()
	 */
	public void focusout();

	/**
	 * Used to access Godot global properties.
	 * @param p_key Property key
	 * @return String value of the property
	 */
	public String getGlobal(String p_key);

	/**
	 * Used to get info about the current rendering system.
	 *
	 * @return A String array with two elements:
	 *         [0] Rendering driver name.
	 *         [1] Rendering method.
	 */
	public String[] getRendererInfo();

	/**
	 * Used to access Godot's editor settings.
	 * @param settingKey Setting key
	 * @return String value of the setting
	 */
	public String getEditorSetting(String settingKey);

	/**
	 * Update the 'key' editor setting with the given data. Must be called on the render thread.
	 * @param key
	 * @param data
	 */
	public void setEditorSetting(String key, Object data);

	/**
	 * Used to access project metadata from the editor settings. Must be accessed on the render thread.
	 * @param section
	 * @param key
	 * @param defaultValue
	 * @return
	 */
	public Object getEditorProjectMetadata(String section, String key, Object defaultValue);

	/**
	 * Set the project metadata to the editor settings. Must be accessed on the render thread.
	 * @param section
	 * @param key
	 * @param data
	 */
	public void setEditorProjectMetadata(String section, String key, Object data);

	/**
	 * Invoke method |p_method| on the Godot object specified by |p_id|
	 * @param p_id Id of the Godot object to invoke
	 * @param p_method Name of the method to invoke
	 * @param p_params Parameters to use for method invocation
	 *
	 * @deprecated Use {@link Callable#call(long, String, Object...)} instead.
	 */
	@Deprecated
	public static void callobject(long p_id, String p_method, Object[] p_params) {
		Callable.call(p_id, p_method, p_params);
	}

	/**
	 * Invoke method |p_method| on the Godot object specified by |p_id| during idle time.
	 * @param p_id Id of the Godot object to invoke
	 * @param p_method Name of the method to invoke
	 * @param p_params Parameters to use for method invocation
	 *
	 * @deprecated Use {@link Callable#callDeferred(long, String, Object...)} instead.
	 */
	@Deprecated
	public static void calldeferred(long p_id, String p_method, Object[] p_params) {
		Callable.callDeferred(p_id, p_method, p_params);
	}

	/**
	 * Forward the results from a permission request.
	 * @see Activity#onRequestPermissionsResult(int, String[], int[])
	 * @param p_permission Request permission
	 * @param p_result True if the permission was granted, false otherwise
	 */
	public void requestPermissionResult(String p_permission, boolean p_result);

	/**
	 * Invoked on the theme light/dark mode change.
	 */
	public void onNightModeChanged();

	/**
	 * Invoked on the hardware keyboard connected/disconnected.
	 */
	public void hardwareKeyboardConnected(boolean connected);

	/**
	 * Invoked on the file picker closed.
	 */
	public void filePickerCallback(boolean p_ok, String[] p_selected_paths);

	/**
	 * Invoked on the GL thread to configure the height of the virtual keyboard.
	 */
	public void setVirtualKeyboardHeight(int p_height);

	/**
	 * Invoked on the GL thread when the {@link GodotRenderer} has been resumed.
	 * @see GodotRenderer#onActivityResumed()
	 */
	public void onRendererResumed();

	/**
	 * Invoked on the GL thread when the {@link GodotRenderer} has been paused.
	 * @see GodotRenderer#onActivityPaused()
	 */
	public void onRendererPaused();

	/**
	 * @return true if input must be dispatched from the render thread. If false, input is
	 * dispatched from the UI thread.
	 */
	public boolean shouldDispatchInputToRenderThread();

	/**
	 * @return the project resource directory
	 */
	public String getProjectResourceDir();

	boolean isEditorHint();

	boolean isProjectManagerHint();

	public boolean providesRenderView();

	public GodotRenderView getRenderView();
}
