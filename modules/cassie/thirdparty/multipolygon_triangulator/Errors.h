#ifndef _ERRORS_H_
#define _ERRORS_H_

#define OUTPUT_ERROR 0

enum ShapeType {
	ERROR_CURVE,
	ERROR_NORM,
	ERROR_MATCH,
	ERROR_LIMIT,
	ERROR_TETGEN,
	ERROR_OTHERS
};

static void errors(int type, char *filename) {
#if OUTPUT_ERROR
	switch (type) {
		case ERROR_CURVE:
			print_line("ERROR: Cannot open curve file!");
			break;
		case ERROR_NORM:
			print_line("ERROR: Cannot open normal file!");
			break;
		case ERROR_MATCH:
			print_line("ERROR: Number of points and number of normals are not match!");
			break;
		case ERROR_LIMIT:
			print_line("ERROR: Number of points exceeds the limit!");
			break;
		case ERROR_TETGEN:
			print_line("ERROR: Tetgen errors happen in edge protection!");
			break;
		case ERROR_OTHERS:
			print_line("ERROR: Got exceptions in main function!");
			break;
		default:
			break;
	}
	String filename_str = String(filename);
	print_line("Curve: " + filename_str);
#endif
}
#endif