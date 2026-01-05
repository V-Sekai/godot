/**************************************************************************/
/*  lexer.h                                                               */
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

#pragma once
#include "token.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace gdscript {

class Lexer {
public:
	explicit Lexer(std::string source);

	std::vector<Token> tokenize();

private:
	void scan_token();
	void scan_string();
	void scan_number();
	void scan_identifier();
	void handle_indent();

	char advance();
	char peek() const;
	char peek_next() const;
	bool match(char expected);
	bool is_at_end() const;
	bool is_digit(char c) const;
	bool is_alpha(char c) const;
	bool is_alphanumeric(char c) const;

	void add_token(TokenType type);
	void add_token(TokenType type, int64_t value);
	void add_token(TokenType type, double value);
	void add_token(TokenType type, const std::string &value);

	void error(const std::string &message);

	std::string m_source;
	std::vector<Token> m_tokens;
	std::vector<int> m_indent_stack; // Track indentation levels

	size_t m_start = 0;
	size_t m_current = 0;
	int m_line = 1;
	int m_column = 1;
	bool m_at_line_start = true;

	static const std::unordered_map<std::string, TokenType> keywords;
};

} // namespace gdscript
