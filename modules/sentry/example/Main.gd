extends Control


func _on_Crash_pressed():
	OS.crash("Crash Event!")


func _on_SendEvent_pressed():
	Sentry.send_event("Test Event!")
