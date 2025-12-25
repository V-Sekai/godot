#ifndef CONFIG_H
#define CONFIG_H

/* This configuration file is custom written for Godot.
 * When updating the library, generate it with configure upstream and compare
 * the contents to see if new options should be backported here.
 */

/* Basic type sizes - assume standard 64-bit platform */
#define SIZEOF_INT 4
#define SIZEOF_LONG 8
#define SIZEOF_LONG_LONG 8
#define SIZEOF_VOID_P 8
#define SIZEOF_SIZE_T 8

/* Endianness */
#ifdef BIG_ENDIAN_ENABLED
#define WORDS_BIGENDIAN
#endif

/* Standard C library features */
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1

/* Socket support */
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_NETINET_IN_H 1
#define HAVE_ARPA_INET_H 1
#define HAVE_NETDB_H 1
#define HAVE_SOCKLEN_T 1
#define HAVE_SYS_UTSNAME_H 1

/* Math functions */
#define HAVE_ISFINITE 1
#define HAVE_ISINF 1
#define HAVE_ISNAN 1

/* Threading */
#define HAVE_PTHREAD_H 1

/* Platform-specific */
#ifdef __APPLE__
#define HAVE_MACH_MACH_H 1
#endif

#ifdef _WIN32
#define HAVE_WINSOCK2_H 1
#define HAVE_WS2TCPIP_H 1
#endif

#endif /* CONFIG_H */

