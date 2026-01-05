/**************************************************************************/
/*  parser.h                                                              */
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
#include "ast.h"
#include "token.h"
#include <memory>
#include <vector>

namespace gdscript {

class Parser {
public:
	explicit Parser(std::vector<Token> tokens);

	Program parse();

private:
	// Function parsing
	FunctionDecl parse_function();
	std::vector<Parameter> parse_parameters();

	// Statement parsing
	StmtPtr parse_statement();
	StmtPtr parse_var_decl(bool is_const);
	StmtPtr parse_if_stmt();
	StmtPtr parse_while_stmt();
	StmtPtr parse_for_stmt();
	StmtPtr parse_return_stmt();
	StmtPtr parse_expr_or_assign_stmt();
	std::vector<StmtPtr> parse_block();

	// Expression parsing (precedence climbing)
	ExprPtr parse_expression();
	ExprPtr parse_or_expression();
	ExprPtr parse_and_expression();
	ExprPtr parse_equality();
	ExprPtr parse_comparison();
	ExprPtr parse_term();
	ExprPtr parse_factor();
	ExprPtr parse_unary();
	ExprPtr parse_call();
	ExprPtr parse_primary();

	// Utilities
	bool match(TokenType type);
	bool match_one_of(std::initializer_list<TokenType> types);
	bool check(TokenType type) const;
	Token advance();
	Token peek() const;
	Token previous() const;
	bool is_at_end() const;
	Token consume(TokenType type, const std::string &message);
	void synchronize();
	void error(const std::string &message);
	void skip_newlines();

	// Type hint parsing
	std::string parse_type_hint(); // Parse optional type hint (e.g., ": int", ": String")
	std::string parse_return_type(); // Parse optional return type (e.g., "-> void")

	// Attribute parsing
	bool parse_attribute(); // Parse attribute (e.g., @export), returns true if @export

	std::vector<Token> m_tokens;
	size_t m_current = 0;
};

} // namespace gdscript
