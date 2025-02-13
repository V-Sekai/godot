/**************************************************************************/
/*  main.c                                                                */
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

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void error(const char *msg) {
	perror(msg);
	exit(1);
}

int main(int argc, char **argv) {
	uint16_t server_port = 8081;
	if (argc > 1) {
		server_port = atoi(argv[1]);
	}

	/*
	 * socket: create TCP stream fd
	 */
	const int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenfd < 0)
		error("Error creating TCP socket");

	const int optval = 1;
	setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
			&optval, sizeof(int));

	struct sockaddr_in serveraddr = {};
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(server_port);

	/*
	 * bind: associate the parent socket with a port
	 */
	if (bind(listenfd, (struct sockaddr *)&serveraddr,
				sizeof(serveraddr)) < 0)
		error("Bind error. Port already in use?");

	/*
	 * listen: listen for incoming TCP connection requests
	 */
	if (listen(listenfd, 5) < 0)
		error("ERROR on listen");

	printf("Listening on port %u\n", server_port);

	/*
	 * main loop: wait for a connection request, echo input line,
	 * then close connection.
	 */
	while (1) {
		/*
		 * accept: wait for a connection request
		 */
		struct sockaddr_in clientaddr;
		socklen_t clientlen = sizeof(clientaddr);
		const int clientfd =
				accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen);
		if (clientfd < 0) {
			printf("ERROR %s on server accept\n", strerror(errno));
			continue;
		}

		/*
		 * inet_ntoa: print who sent the message
		 */
		char *ipstr = inet_ntoa(clientaddr.sin_addr);
		if (ipstr == NULL)
			error("ERROR on inet_ntoa\n");
		printf("Server established connection with %s\n",
				ipstr);

		/*
		 * read: read input string from the client
		 */
		char buffer[8192];
		const ssize_t rb = read(clientfd, buffer, sizeof(buffer));
		if (rb < 0) {
			printf("ERROR %s reading from socket\n", strerror(errno));
			close(clientfd);
		} else if (rb == 0) {
			close(clientfd);
			continue;
		}
		printf("Server received %u bytes: %.*s",
				(unsigned)rb, (int)rb, buffer);

		/*
		 * write: echo the input string back to the client
		 */
		const ssize_t wb = write(clientfd, buffer, rb);
		if (wb < 0) {
			printf("ERROR %s writing to socket\n", strerror(errno));
			close(clientfd);
			continue;
		}

		close(clientfd);
	}
}
