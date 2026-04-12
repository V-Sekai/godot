/*
 * Copyright (c) 2026 K. S. Ernest (iFire) Lee
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

// Android keychain backend using the AndroidKeyStore provider via JNI.
//
// Secrets are AES-GCM encrypted with a hardware-backed key stored in
// AndroidKeyStore and the resulting ciphertext is persisted in
// SharedPreferences.  The preference file is MODE_PRIVATE, readable
// only by this application.
//
// Requires API level 23+ (Android 6.0 Marshmallow) which is below
// Godot 4's minimum of API 24.

#include "keychain.h"

#include <jni.h>
#include <string>
#include <vector>

// Godot provides a thread-local JNIEnv accessor.
#include "platform/android/thread_jandroid.h"
#include "platform/android/java_godot_wrapper.h"
#include "platform/android/os_android.h"

namespace {

// Preference file name — private to the application.
static const char *PREFS_NAME = "godot_keychain";

// Convert std::string → jstring (UTF-8).
jstring to_jstring(JNIEnv *env, const std::string &s) {
    return env->NewStringUTF(s.c_str());
}

// Convert jstring → std::string.  Returns empty string on null.
std::string from_jstring(JNIEnv *env, jstring js) {
    if (!js) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

// Build the SharedPreferences key from the keychain tuple.
std::string make_prefs_key(const std::string &package,
                           const std::string &service,
                           const std::string &user) {
    return package + "." + service + "/" + user;
}

// Alias used inside AndroidKeyStore for our AES wrapping key.
std::string make_ks_alias(const std::string &package,
                          const std::string &service) {
    return package + "." + service + ".keychain";
}

// Check for and clear a pending Java exception, returning its message.
bool check_exception(JNIEnv *env, keychain::Error &err) {
    if (!env->ExceptionCheck()) {
        return false;
    }
    jthrowable ex = env->ExceptionOccurred();
    env->ExceptionClear();

    jclass cls = env->GetObjectClass(ex);
    jmethodID getMessage = env->GetMethodID(cls, "getMessage",
                                            "()Ljava/lang/String;");
    jstring msg = (jstring)env->CallObjectMethod(ex, getMessage);

    err.type = keychain::ErrorType::GenericError;
    err.message = msg ? from_jstring(env, msg) : "Unknown Java exception";
    err.code = -1;

    env->DeleteLocalRef(msg);
    env->DeleteLocalRef(cls);
    env->DeleteLocalRef(ex);
    return true;
}

// Ensure the AndroidKeyStore contains an AES-256-GCM key for this alias.
// Does nothing if the key already exists.
bool ensure_aes_key(JNIEnv *env, const std::string &alias,
                    keychain::Error &err) {
    // KeyStore ks = KeyStore.getInstance("AndroidKeyStore");
    jclass ksClass = env->FindClass("java/security/KeyStore");
    jmethodID getInstance = env->GetStaticMethodID(
        ksClass, "getInstance",
        "(Ljava/lang/String;)Ljava/security/KeyStore;");
    jstring provider = to_jstring(env, "AndroidKeyStore");
    jobject ks = env->CallStaticObjectMethod(ksClass, getInstance, provider);
    if (check_exception(env, err)) {
        return false;
    }

    // ks.load(null);
    jmethodID load = env->GetMethodID(
        ksClass, "load", "(Ljava/security/KeyStore$LoadStoreParameter;)V");
    env->CallVoidMethod(ks, load, nullptr);
    if (check_exception(env, err)) {
        return false;
    }

    // if (ks.containsAlias(alias)) return true;
    jmethodID containsAlias = env->GetMethodID(
        ksClass, "containsAlias", "(Ljava/lang/String;)Z");
    jstring jAlias = to_jstring(env, alias);
    jboolean exists = env->CallBooleanMethod(ks, containsAlias, jAlias);
    if (check_exception(env, err)) {
        return false;
    }
    if (exists) {
        env->DeleteLocalRef(jAlias);
        env->DeleteLocalRef(ks);
        env->DeleteLocalRef(provider);
        env->DeleteLocalRef(ksClass);
        return true;
    }

    // KeyGenParameterSpec.Builder builder =
    //     new KeyGenParameterSpec.Builder(alias,
    //         KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT);
    jclass builderClass = env->FindClass(
        "android/security/keystore/KeyGenParameterSpec$Builder");
    jmethodID builderCtor = env->GetMethodID(
        builderClass, "<init>", "(Ljava/lang/String;I)V");
    // PURPOSE_ENCRYPT=1, PURPOSE_DECRYPT=2
    jobject builder = env->NewObject(builderClass, builderCtor, jAlias, 3);
    if (check_exception(env, err)) {
        return false;
    }

    // builder.setBlockModes("GCM");
    jmethodID setBlockModes = env->GetMethodID(
        builderClass, "setBlockModes",
        "([Ljava/lang/String;)"
        "Landroid/security/keystore/KeyGenParameterSpec$Builder;");
    jstring gcm = to_jstring(env, "GCM");
    jobjectArray modes = env->NewObjectArray(
        1, env->FindClass("java/lang/String"), gcm);
    env->CallObjectMethod(builder, setBlockModes, modes);

    // builder.setEncryptionPaddings("NoPadding");
    jmethodID setEncPad = env->GetMethodID(
        builderClass, "setEncryptionPaddings",
        "([Ljava/lang/String;)"
        "Landroid/security/keystore/KeyGenParameterSpec$Builder;");
    jstring noPad = to_jstring(env, "NoPadding");
    jobjectArray pads = env->NewObjectArray(
        1, env->FindClass("java/lang/String"), noPad);
    env->CallObjectMethod(builder, setEncPad, pads);

    // builder.setKeySize(256);
    jmethodID setKeySize = env->GetMethodID(
        builderClass, "setKeySize",
        "(I)Landroid/security/keystore/KeyGenParameterSpec$Builder;");
    env->CallObjectMethod(builder, setKeySize, 256);

    // KeyGenParameterSpec spec = builder.build();
    jmethodID buildSpec = env->GetMethodID(
        builderClass, "build",
        "()Landroid/security/keystore/KeyGenParameterSpec;");
    jobject spec = env->CallObjectMethod(builder, buildSpec);
    if (check_exception(env, err)) {
        return false;
    }

    // KeyGenerator kg = KeyGenerator.getInstance("AES", "AndroidKeyStore");
    jclass kgClass = env->FindClass("javax/crypto/KeyGenerator");
    jmethodID kgGetInstance = env->GetStaticMethodID(
        kgClass, "getInstance",
        "(Ljava/lang/String;Ljava/lang/String;)"
        "Ljavax/crypto/KeyGenerator;");
    jstring aes = to_jstring(env, "AES");
    jobject kg = env->CallStaticObjectMethod(
        kgClass, kgGetInstance, aes, provider);
    if (check_exception(env, err)) {
        return false;
    }

    // kg.init(spec);
    jmethodID kgInit = env->GetMethodID(
        kgClass, "init",
        "(Ljava/security/spec/AlgorithmParameterSpec;)V");
    env->CallVoidMethod(kg, kgInit, spec);
    if (check_exception(env, err)) {
        return false;
    }

    // kg.generateKey();
    jmethodID generateKey = env->GetMethodID(
        kgClass, "generateKey", "()Ljavax/crypto/SecretKey;");
    env->CallObjectMethod(kg, generateKey);
    if (check_exception(env, err)) {
        return false;
    }

    // Cleanup local refs.
    env->DeleteLocalRef(kgClass);
    env->DeleteLocalRef(kg);
    env->DeleteLocalRef(aes);
    env->DeleteLocalRef(spec);
    env->DeleteLocalRef(builderClass);
    env->DeleteLocalRef(builder);
    env->DeleteLocalRef(modes);
    env->DeleteLocalRef(gcm);
    env->DeleteLocalRef(pads);
    env->DeleteLocalRef(noPad);
    env->DeleteLocalRef(jAlias);
    env->DeleteLocalRef(ks);
    env->DeleteLocalRef(provider);
    env->DeleteLocalRef(ksClass);

    return true;
}

// Retrieve the SecretKey from AndroidKeyStore.
jobject get_secret_key(JNIEnv *env, const std::string &alias,
                       keychain::Error &err) {
    jclass ksClass = env->FindClass("java/security/KeyStore");
    jmethodID getInstance = env->GetStaticMethodID(
        ksClass, "getInstance",
        "(Ljava/lang/String;)Ljava/security/KeyStore;");
    jstring provider = to_jstring(env, "AndroidKeyStore");
    jobject ks = env->CallStaticObjectMethod(ksClass, getInstance, provider);
    if (check_exception(env, err)) {
        return nullptr;
    }

    jmethodID load = env->GetMethodID(
        ksClass, "load", "(Ljava/security/KeyStore$LoadStoreParameter;)V");
    env->CallVoidMethod(ks, load, nullptr);
    if (check_exception(env, err)) {
        return nullptr;
    }

    // KeyStore.Entry entry = ks.getEntry(alias, null);
    jmethodID getEntry = env->GetMethodID(
        ksClass, "getEntry",
        "(Ljava/lang/String;Ljava/security/KeyStore$ProtectionParameter;)"
        "Ljava/security/KeyStore$Entry;");
    jstring jAlias = to_jstring(env, alias);
    jobject entry = env->CallObjectMethod(ks, getEntry, jAlias, nullptr);
    if (check_exception(env, err) || !entry) {
        err.type = keychain::ErrorType::NotFound;
        err.message = "Key alias not found in AndroidKeyStore.";
        return nullptr;
    }

    // SecretKey key = ((KeyStore.SecretKeyEntry) entry).getSecretKey();
    jclass skeClass = env->FindClass(
        "java/security/KeyStore$SecretKeyEntry");
    jmethodID getSecretKey = env->GetMethodID(
        skeClass, "getSecretKey", "()Ljavax/crypto/SecretKey;");
    jobject secretKey = env->CallObjectMethod(entry, getSecretKey);

    env->DeleteLocalRef(skeClass);
    env->DeleteLocalRef(entry);
    env->DeleteLocalRef(jAlias);
    env->DeleteLocalRef(ks);
    env->DeleteLocalRef(provider);
    env->DeleteLocalRef(ksClass);

    return secretKey;
}

// Get the application SharedPreferences via the Activity context.
jobject get_shared_prefs(JNIEnv *env, keychain::Error &err) {
    GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
    jobject activity = godot_java->get_activity();
    if (!activity) {
        err.type = keychain::ErrorType::GenericError;
        err.message = "Cannot obtain Android Activity.";
        err.code = -1;
        return nullptr;
    }

    jclass ctxClass = env->FindClass("android/content/Context");
    jmethodID getSharedPrefs = env->GetMethodID(
        ctxClass, "getSharedPreferences",
        "(Ljava/lang/String;I)Landroid/content/SharedPreferences;");
    jstring prefsName = to_jstring(env, PREFS_NAME);
    // MODE_PRIVATE = 0
    jobject prefs = env->CallObjectMethod(activity, getSharedPrefs,
                                          prefsName, 0);
    if (check_exception(env, err)) {
        return nullptr;
    }

    env->DeleteLocalRef(prefsName);
    env->DeleteLocalRef(ctxClass);
    return prefs;
}

// Base64 encode/decode helpers using android.util.Base64.
std::string base64_encode(JNIEnv *env, const std::vector<uint8_t> &data) {
    jclass b64 = env->FindClass("android/util/Base64");
    jmethodID encode = env->GetStaticMethodID(
        b64, "encodeToString", "([BI)Ljava/lang/String;");
    jbyteArray arr = env->NewByteArray(data.size());
    env->SetByteArrayRegion(arr, 0, data.size(),
                            reinterpret_cast<const jbyte *>(data.data()));
    // NO_WRAP = 2
    jstring result = (jstring)env->CallStaticObjectMethod(
        b64, encode, arr, 2);
    std::string s = from_jstring(env, result);
    env->DeleteLocalRef(result);
    env->DeleteLocalRef(arr);
    env->DeleteLocalRef(b64);
    return s;
}

std::vector<uint8_t> base64_decode(JNIEnv *env, const std::string &s) {
    jclass b64 = env->FindClass("android/util/Base64");
    jmethodID decode = env->GetStaticMethodID(
        b64, "decode", "(Ljava/lang/String;I)[B");
    jstring js = to_jstring(env, s);
    // NO_WRAP = 2
    jbyteArray arr = (jbyteArray)env->CallStaticObjectMethod(
        b64, decode, js, 2);
    jsize len = env->GetArrayLength(arr);
    std::vector<uint8_t> result(len);
    env->GetByteArrayRegion(arr, 0, len,
                            reinterpret_cast<jbyte *>(result.data()));
    env->DeleteLocalRef(arr);
    env->DeleteLocalRef(js);
    env->DeleteLocalRef(b64);
    return result;
}

} // namespace

namespace keychain {

void setPassword(const std::string &package, const std::string &service,
                 const std::string &user, const std::string &password,
                 Error &err) {
    err = Error{};
    JNIEnv *env = get_jni_env();
    if (!env) {
        err.type = ErrorType::GenericError;
        err.message = "JNI environment not available.";
        err.code = -1;
        return;
    }

    const std::string alias = make_ks_alias(package, service);
    if (!ensure_aes_key(env, alias, err)) {
        return;
    }

    jobject secretKey = get_secret_key(env, alias, err);
    if (!secretKey) {
        return;
    }

    // Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    jclass cipherClass = env->FindClass("javax/crypto/Cipher");
    jmethodID cipherGetInstance = env->GetStaticMethodID(
        cipherClass, "getInstance",
        "(Ljava/lang/String;)Ljavax/crypto/Cipher;");
    jstring transform = to_jstring(env, "AES/GCM/NoPadding");
    jobject cipher = env->CallStaticObjectMethod(
        cipherClass, cipherGetInstance, transform);
    if (check_exception(env, err)) {
        return;
    }

    // cipher.init(Cipher.ENCRYPT_MODE, secretKey);
    jmethodID initCipher = env->GetMethodID(
        cipherClass, "init", "(ILjava/security/Key;)V");
    // ENCRYPT_MODE = 1
    env->CallVoidMethod(cipher, initCipher, 1, secretKey);
    if (check_exception(env, err)) {
        return;
    }

    // byte[] iv = cipher.getIV();
    jmethodID getIV = env->GetMethodID(cipherClass, "getIV", "()[B");
    jbyteArray ivArr = (jbyteArray)env->CallObjectMethod(cipher, getIV);
    jsize ivLen = env->GetArrayLength(ivArr);
    std::vector<uint8_t> iv(ivLen);
    env->GetByteArrayRegion(ivArr, 0, ivLen,
                            reinterpret_cast<jbyte *>(iv.data()));

    // byte[] encrypted = cipher.doFinal(password.getBytes("UTF-8"));
    jmethodID doFinal = env->GetMethodID(
        cipherClass, "doFinal", "([B)[B");
    jbyteArray plainArr = env->NewByteArray(password.size());
    env->SetByteArrayRegion(plainArr, 0, password.size(),
                            reinterpret_cast<const jbyte *>(password.data()));
    jbyteArray encArr = (jbyteArray)env->CallObjectMethod(
        cipher, doFinal, plainArr);
    if (check_exception(env, err)) {
        return;
    }

    jsize encLen = env->GetArrayLength(encArr);
    std::vector<uint8_t> encrypted(encLen);
    env->GetByteArrayRegion(encArr, 0, encLen,
                            reinterpret_cast<jbyte *>(encrypted.data()));

    // Store IV + ciphertext as base64 in SharedPreferences.
    // Format: base64(iv) + ":" + base64(ciphertext)
    std::string stored = base64_encode(env, iv) + ":" +
                         base64_encode(env, encrypted);

    jobject prefs = get_shared_prefs(env, err);
    if (!prefs) {
        return;
    }

    // SharedPreferences.Editor editor = prefs.edit();
    jclass prefsClass = env->FindClass("android/content/SharedPreferences");
    jmethodID edit = env->GetMethodID(
        prefsClass, "edit",
        "()Landroid/content/SharedPreferences$Editor;");
    jobject editor = env->CallObjectMethod(prefs, edit);

    // editor.putString(key, stored);
    jclass editorClass = env->FindClass(
        "android/content/SharedPreferences$Editor");
    jmethodID putString = env->GetMethodID(
        editorClass, "putString",
        "(Ljava/lang/String;Ljava/lang/String;)"
        "Landroid/content/SharedPreferences$Editor;");
    jstring jKey = to_jstring(env, make_prefs_key(package, service, user));
    jstring jVal = to_jstring(env, stored);
    env->CallObjectMethod(editor, putString, jKey, jVal);

    // editor.apply();
    jmethodID apply = env->GetMethodID(editorClass, "apply", "()V");
    env->CallVoidMethod(editor, apply);

    // Cleanup.
    env->DeleteLocalRef(jVal);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(editorClass);
    env->DeleteLocalRef(editor);
    env->DeleteLocalRef(prefsClass);
    env->DeleteLocalRef(prefs);
    env->DeleteLocalRef(encArr);
    env->DeleteLocalRef(plainArr);
    env->DeleteLocalRef(ivArr);
    env->DeleteLocalRef(cipher);
    env->DeleteLocalRef(transform);
    env->DeleteLocalRef(cipherClass);
    env->DeleteLocalRef(secretKey);
}

std::string getPassword(const std::string &package, const std::string &service,
                        const std::string &user, Error &err) {
    err = Error{};
    JNIEnv *env = get_jni_env();
    if (!env) {
        err.type = ErrorType::GenericError;
        err.message = "JNI environment not available.";
        err.code = -1;
        return {};
    }

    // Read from SharedPreferences.
    jobject prefs = get_shared_prefs(env, err);
    if (!prefs) {
        return {};
    }

    jclass prefsClass = env->FindClass("android/content/SharedPreferences");
    jmethodID getString = env->GetMethodID(
        prefsClass, "getString",
        "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    jstring jKey = to_jstring(env, make_prefs_key(package, service, user));
    jstring jVal = (jstring)env->CallObjectMethod(
        prefs, getString, jKey, nullptr);

    if (!jVal) {
        err.type = ErrorType::NotFound;
        err.message = "Password not found.";
        err.code = -1;
        env->DeleteLocalRef(jKey);
        env->DeleteLocalRef(prefsClass);
        env->DeleteLocalRef(prefs);
        return {};
    }

    std::string stored = from_jstring(env, jVal);
    env->DeleteLocalRef(jVal);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(prefsClass);
    env->DeleteLocalRef(prefs);

    // Parse "base64(iv):base64(ciphertext)".
    auto sep = stored.find(':');
    if (sep == std::string::npos) {
        err.type = ErrorType::GenericError;
        err.message = "Malformed stored credential.";
        err.code = -1;
        return {};
    }

    std::vector<uint8_t> iv = base64_decode(env, stored.substr(0, sep));
    std::vector<uint8_t> encrypted = base64_decode(env, stored.substr(sep + 1));

    // Get the key.
    const std::string alias = make_ks_alias(package, service);
    jobject secretKey = get_secret_key(env, alias, err);
    if (!secretKey) {
        return {};
    }

    // GCMParameterSpec spec = new GCMParameterSpec(128, iv);
    jclass gcmSpecClass = env->FindClass(
        "javax/crypto/spec/GCMParameterSpec");
    jmethodID gcmCtor = env->GetMethodID(
        gcmSpecClass, "<init>", "(I[B)V");
    jbyteArray ivArr = env->NewByteArray(iv.size());
    env->SetByteArrayRegion(ivArr, 0, iv.size(),
                            reinterpret_cast<const jbyte *>(iv.data()));
    jobject gcmSpec = env->NewObject(gcmSpecClass, gcmCtor, 128, ivArr);

    // Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    jclass cipherClass = env->FindClass("javax/crypto/Cipher");
    jmethodID cipherGetInstance = env->GetStaticMethodID(
        cipherClass, "getInstance",
        "(Ljava/lang/String;)Ljavax/crypto/Cipher;");
    jstring transform = to_jstring(env, "AES/GCM/NoPadding");
    jobject cipher = env->CallStaticObjectMethod(
        cipherClass, cipherGetInstance, transform);
    if (check_exception(env, err)) {
        return {};
    }

    // cipher.init(Cipher.DECRYPT_MODE, secretKey, gcmSpec);
    jmethodID initCipher = env->GetMethodID(
        cipherClass, "init",
        "(ILjava/security/Key;Ljava/security/spec/AlgorithmParameterSpec;)V");
    // DECRYPT_MODE = 2
    env->CallVoidMethod(cipher, initCipher, 2, secretKey, gcmSpec);
    if (check_exception(env, err)) {
        return {};
    }

    // byte[] decrypted = cipher.doFinal(encrypted);
    jmethodID doFinal = env->GetMethodID(
        cipherClass, "doFinal", "([B)[B");
    jbyteArray encArr = env->NewByteArray(encrypted.size());
    env->SetByteArrayRegion(encArr, 0, encrypted.size(),
                            reinterpret_cast<const jbyte *>(encrypted.data()));
    jbyteArray decArr = (jbyteArray)env->CallObjectMethod(
        cipher, doFinal, encArr);
    if (check_exception(env, err)) {
        return {};
    }

    jsize decLen = env->GetArrayLength(decArr);
    std::string password(decLen, '\0');
    env->GetByteArrayRegion(decArr, 0, decLen,
                            reinterpret_cast<jbyte *>(&password[0]));

    env->DeleteLocalRef(decArr);
    env->DeleteLocalRef(encArr);
    env->DeleteLocalRef(cipher);
    env->DeleteLocalRef(transform);
    env->DeleteLocalRef(cipherClass);
    env->DeleteLocalRef(gcmSpec);
    env->DeleteLocalRef(ivArr);
    env->DeleteLocalRef(gcmSpecClass);
    env->DeleteLocalRef(secretKey);

    return password;
}

void deletePassword(const std::string &package, const std::string &service,
                    const std::string &user, Error &err) {
    err = Error{};
    JNIEnv *env = get_jni_env();
    if (!env) {
        err.type = ErrorType::GenericError;
        err.message = "JNI environment not available.";
        err.code = -1;
        return;
    }

    jobject prefs = get_shared_prefs(env, err);
    if (!prefs) {
        return;
    }

    // Check if the key exists.
    jclass prefsClass = env->FindClass("android/content/SharedPreferences");
    jmethodID contains = env->GetMethodID(
        prefsClass, "contains", "(Ljava/lang/String;)Z");
    jstring jKey = to_jstring(env, make_prefs_key(package, service, user));
    jboolean exists = env->CallBooleanMethod(prefs, contains, jKey);

    if (!exists) {
        err.type = ErrorType::NotFound;
        err.message = "Password not found.";
        err.code = -1;
        env->DeleteLocalRef(jKey);
        env->DeleteLocalRef(prefsClass);
        env->DeleteLocalRef(prefs);
        return;
    }

    // editor.remove(key).apply();
    jmethodID edit = env->GetMethodID(
        prefsClass, "edit",
        "()Landroid/content/SharedPreferences$Editor;");
    jobject editor = env->CallObjectMethod(prefs, edit);

    jclass editorClass = env->FindClass(
        "android/content/SharedPreferences$Editor");
    jmethodID remove = env->GetMethodID(
        editorClass, "remove",
        "(Ljava/lang/String;)"
        "Landroid/content/SharedPreferences$Editor;");
    env->CallObjectMethod(editor, remove, jKey);

    jmethodID apply = env->GetMethodID(editorClass, "apply", "()V");
    env->CallVoidMethod(editor, apply);

    env->DeleteLocalRef(editorClass);
    env->DeleteLocalRef(editor);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(prefsClass);
    env->DeleteLocalRef(prefs);
}

} // namespace keychain
