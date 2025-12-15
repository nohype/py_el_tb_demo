import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

import xlwings as xw
from antlr4 import ParserRuleContext
from antlr4.tree.Tree import TerminalNode

from antlr_utils import (
    AntlrParserObjects,
    ELTreeWalker,
    ctx_start_token,
    ctx_stop_token,
    get_outermost_parent_scope,
    get_source_text_ctx,
    get_source_text_terminal,
    token_start_index,
    token_stop_index,
)
from el_antlr_helpers import (
    ELDeclarationItemContent,
    is_buy_order,
    is_buy_to_cover_order,
    is_sell_order,
    is_sell_short_order,
)
from el_ast import (
    CaseBlock,
    CodeBlock,
    CodeLocation,
    OrderDirection,
    SlPtDirective,
    SwitchBlock,
    SwitchBlockContainer,
    Variable,
    VariableDeclBlock,
    merge_overlapping_code_locations,
)
from generated.elantlr.EasyLanguageParser import EasyLanguageParser
from toolbox_handler import ToolboxHandler


def _read_file_to_string(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8-sig")


class ELDocument:
    def __init__(self, code: str = "", file_path: Optional[Union[str, Path]] = None):
        self._file_path = Path(file_path) if file_path else Path()
        self._raw_code = ""
        self._parser_objects: Optional[AntlrParserObjects] = None

        self._variable_declarations: dict[str, list[VariableDeclBlock]] = {}
        self._switch_blocks = SwitchBlockContainer()
        self._order_blocks: dict[OrderDirection, list[CodeBlock]] = {}
        self._sl_pt_blocks: dict[SlPtDirective, list[CodeBlock]] = {}
        self._set_exit_on_close_blocks: list[CodeBlock] = []
        self._terminals: list[CodeBlock] = []

        if code:
            self._parser_objects = AntlrParserObjects.from_string(code)
            self._raw_code = code
        else:
            self._raw_code = ""

    @staticmethod
    def from_file(file_path: Union[str, Path]) -> Optional["ELDocument"]:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None

        code = _read_file_to_string(p)
        if not code:
            return None

        return ELDocument(code=code, file_path=p)

    def code(self) -> str:
        return self._raw_code

    def file_path(self) -> Path:
        return self._file_path

    def loc_from(self, obj: object) -> CodeLocation:
        if not self._parser_objects:
            return CodeLocation()

        return self._parser_objects.loc_from(obj)

    def get_char_at(self, pos: int) -> str:
        return self._raw_code[pos]

    def get_stream_char_at(self, pos: int) -> str:
        return self._raw_code[pos]

    def reset(self) -> None:
        self._clear_caches()
        self._parser_objects = AntlrParserObjects.from_string(self._raw_code)

    def _scoped_code_block(self, ctx: ParserRuleContext) -> CodeBlock:
        enclosing = get_outermost_parent_scope(ctx)
        loc = self.loc_from(enclosing) if enclosing else self.loc_from(ctx)
        code = get_source_text_ctx(enclosing, self._raw_code) if enclosing else get_source_text_ctx(ctx, self._raw_code)
        return CodeBlock(loc, code)

    def variable_declarations(self, declaration_type: str) -> list[VariableDeclBlock]:
        if not self._variable_declarations:
            self._variable_declarations = self._extract_variable_declarations()

        key = declaration_type.lower()
        return self._variable_declarations.get(key, [])

    def all_variable_declarations(self) -> dict[str, list[VariableDeclBlock]]:
        if not self._variable_declarations:
            self._variable_declarations = self._extract_variable_declarations()

        return self._variable_declarations

    def switch_blocks(self) -> SwitchBlockContainer:
        if self._switch_blocks.empty():
            self._switch_blocks = self._extract_switch_blocks()

        return self._switch_blocks

    def order_blocks(self) -> dict[OrderDirection, list[CodeBlock]]:
        if not self._order_blocks:
            self._order_blocks = self._extract_order_blocks()

        return self._order_blocks

    def sl_pt_blocks(self) -> dict[SlPtDirective, list[CodeBlock]]:
        if not self._sl_pt_blocks:
            self._sl_pt_blocks = self._extract_sl_pt_blocks()

        return self._sl_pt_blocks

    def set_exit_on_close_blocks(self) -> list[CodeBlock]:
        if not self._set_exit_on_close_blocks:
            self._set_exit_on_close_blocks = self._extract_set_exit_on_close_blocks()

        return self._set_exit_on_close_blocks

    def variable_declarations_merged(self, declaration_type: str) -> dict[str, Variable]:
        results: dict[str, Variable] = {}
        decl_blocks = self.variable_declarations(declaration_type)
        for block in decl_blocks:
            for var in block:
                if var.name not in results:
                    results[var.name] = var
        return results

    def terminals(self) -> list[CodeBlock]:
        if not self._terminals:
            self._terminals = self._extract_terminals()

        return self._terminals

    def is_empty(self) -> bool:
        return self._raw_code == ""

    def is_valid(self) -> bool:
        return self._parser_objects is not None

    def is_valid_location(self, loc: CodeLocation) -> bool:
        return (
            not self.is_empty()
            and loc.is_valid()
            and loc.firstPos < len(self._raw_code)
            and loc.lastPos < len(self._raw_code)
        )

    def code_with_locations_removed(self, locations: list[CodeLocation]) -> str:
        cleaned = self._raw_code
        sorted_locs = sorted(locations)
        for loc in sorted_locs:
            if not loc.is_valid():
                continue

            cleaned = cleaned[: loc.firstPos] + cleaned[loc.firstPos + loc.length() :]

        return cleaned

    def _parse_tree(self) -> EasyLanguageParser.StartContext:
        if not self._parser_objects:
            raise RuntimeError("Parser objects are not initialized")

        try:
            if hasattr(self._parser_objects.tokens, "seek"):
                self._parser_objects.tokens.seek(0)
        except Exception:
            pass

        try:
            self._parser_objects.parser.reset()
        except Exception:
            pass

        return self._parser_objects.parser.start()

    def _extract_variable_declarations(self) -> dict[str, list[VariableDeclBlock]]:
        if not self._parser_objects:
            return {}

        tree = self._parse_tree()
        results: dict[str, list[VariableDeclBlock]] = {}
        program = tree.program()

        def process_declaration(
            items: list[ELDeclarationItemContent],
            category: str,
            block_loc: CodeLocation,
        ) -> None:
            block = VariableDeclBlock()
            block.loc = block_loc
            for item in items:
                type_ctx = item.type_ref()
                type_str = get_source_text_ctx(type_ctx, self._raw_code) if type_ctx else ""
                ident_node = item.IDENT()
                name_str = ident_node.getText() if ident_node else ""
                expr_ctx = item.expr()
                value_str = get_source_text_ctx(expr_ctx, self._raw_code) if expr_ctx else ""
                block.add(Variable(type_str, name_str, value_str, self.loc_from(item)))

            results.setdefault(category, []).append(block)

        for decl in program.declaration():
            id_ctx = decl.input_decl()
            if id_ctx:
                items = id_ctx.input_item()
                process_declaration(items, "inputs", self.loc_from(id_ctx))
                continue

            vd_ctx = decl.var_decl()
            if vd_ctx:
                items = vd_ctx.var_item()
                process_declaration(items, "vars", self.loc_from(vd_ctx))
                continue

            cd_ctx = decl.const_decl()
            if cd_ctx:
                items = cd_ctx.const_item()
                process_declaration(items, "consts", self.loc_from(cd_ctx))
                continue

        return results

    def _extract_switch_blocks(self) -> SwitchBlockContainer:
        blocks = SwitchBlockContainer()
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Switch_stmtContext) -> None:
            sb = self._extract_switch_block(ctx)
            blocks.add(sb)

        extractor.register_ctx_handler(EasyLanguageParser.Switch_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return blocks

    def _extract_switch_block(self, ctx: EasyLanguageParser.Switch_stmtContext) -> SwitchBlock:
        sb = SwitchBlock()
        sb.loc = self.loc_from(ctx)
        expr_ctx = ctx.expr()
        sb.switchVar = get_source_text_ctx(expr_ctx, self._raw_code) if expr_ctx else ""

        body = ctx.switch_body()
        if body:
            for sec in body.switch_section():
                section_code = ""
                section_loc = CodeLocation()
                stmts = sec.statement()
                if stmts:
                    first = stmts[0]
                    last = stmts[-1]
                    first_start = token_start_index(ctx_start_token(first))
                    last_stop = token_stop_index(ctx_stop_token(last))
                    if first_start is not None and last_stop is not None:
                        if 0 <= first_start <= last_stop < len(self._raw_code):
                            section_code = self._raw_code[first_start : last_stop + 1]

                    first_loc = self.loc_from(first)
                    last_loc = self.loc_from(last)
                    if first_loc.is_valid() and last_loc.is_valid():
                        section_loc = CodeLocation(first_loc.firstPos, last_loc.lastPos)

                    if len(stmts) == 1:
                        blk = stmts[0].block()
                        if blk:
                            begin_term = blk.BEGIN()
                            end_term = blk.END()
                            begin_sym = begin_term.getSymbol() if begin_term else None
                            end_sym = end_term.getSymbol() if end_term else None
                            begin_stop = token_stop_index(begin_sym)
                            end_start = token_start_index(end_sym)
                            if begin_stop is not None and end_start is not None:
                                start_idx = begin_stop + 1
                                end_idx = end_start - 1
                                if 0 <= start_idx <= end_idx < len(self._raw_code):
                                    section_code = self._raw_code[start_idx : end_idx + 1]
                                    section_loc = CodeLocation(start_idx, end_idx)

                                    while (
                                        section_code
                                        and section_code[0] in ("\n", "\r")
                                        and section_loc.firstPos < section_loc.lastPos
                                    ):
                                        section_code = section_code[1:]
                                        section_loc.firstPos += 1

                                    while (
                                        section_code
                                        and section_code[-1] in ("\n", "\r")
                                        and section_loc.firstPos < section_loc.lastPos
                                    ):
                                        section_code = section_code[:-1]
                                        section_loc.lastPos -= 1

                if sec.CASE():
                    labels = sec.case_labels()
                    if labels:
                        for item in labels.case_label_item():
                            cb = CaseBlock(section_loc, get_source_text_ctx(item, self._raw_code), section_code)
                            sb.add(cb)
                elif sec.DEFAULT():
                    cb = CaseBlock(section_loc, "default", section_code)
                    sb.add(cb)

        return sb

    def _extract_order_blocks(self) -> dict[OrderDirection, list[CodeBlock]]:
        results: dict[OrderDirection, list[CodeBlock]] = {}
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Order_stmtContext) -> None:
            cb = self._scoped_code_block(ctx)
            side = ctx.order_side()
            if is_buy_order(side):
                results.setdefault(OrderDirection.LongEntry, []).append(cb)
            elif is_sell_short_order(side):
                results.setdefault(OrderDirection.ShortEntry, []).append(cb)
            elif is_sell_order(side):
                results.setdefault(OrderDirection.LongExit, []).append(cb)
            elif is_buy_to_cover_order(side):
                results.setdefault(OrderDirection.ShortExit, []).append(cb)

        extractor.register_ctx_handler(EasyLanguageParser.Order_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        for k, vec in results.items():
            merge_overlapping_code_locations(vec)

        return results

    def _extract_sl_pt_blocks(self) -> dict[SlPtDirective, list[CodeBlock]]:
        results: dict[SlPtDirective, list[CodeBlock]] = {}
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Sl_pt_directive_stmtContext) -> None:
            cb = self._scoped_code_block(ctx)
            if ctx.setStopLossStmt():
                results.setdefault(SlPtDirective.SetStopLoss, []).append(cb)
            elif ctx.setProfitTargetStmt():
                results.setdefault(SlPtDirective.SetProfitTarget, []).append(cb)
            else:
                results.setdefault(SlPtDirective.SetContractOrPosition, []).append(cb)

        extractor.register_ctx_handler(EasyLanguageParser.Sl_pt_directive_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        for k, vec in results.items():
            merge_overlapping_code_locations(vec)

        return results

    def _extract_set_exit_on_close_blocks(self) -> list[CodeBlock]:
        results: list[CodeBlock] = []
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.SetExitOnCloseStmtContext) -> None:
            results.append(self._scoped_code_block(ctx))

        extractor.register_ctx_handler(EasyLanguageParser.SetExitOnCloseStmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return results

    def _extract_terminals(self) -> list[CodeBlock]:
        results: list[CodeBlock] = []
        extractor = ELTreeWalker()

        def handler(node: TerminalNode) -> None:
            loc = self.loc_from(node)
            code = get_source_text_terminal(node, self._raw_code)
            results.append(CodeBlock(loc, code))

        extractor.register_terminal_handler(handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return results

    def _clear_caches(self) -> None:
        self._variable_declarations.clear()
        self._switch_blocks.clear()
        self._order_blocks.clear()
        self._sl_pt_blocks.clear()
        self._set_exit_on_close_blocks.clear()


def test_vars(el_code: str, doc: ELDocument):
    all_vars = doc.all_variable_declarations()
    for category, decl_blocks in all_vars.items():
        print(f"{category}:")
        for block in decl_blocks:
            for var in block:
                print(
                    f"\t{var.dataType} {var.name}({var.value}) [raw code: `{el_code[var.loc.firstPos : var.loc.lastPos + 1]}`]"
                )


def test_switches(el_code: str, doc: ELDocument):
    all_switch_blocks = doc.switch_blocks()
    for sw_block in all_switch_blocks:
        print("=" * 100)
        print(f"switch {sw_block.switchVar}:")
        for case_block in sw_block:
            print(f"\t{case_block.caseValue}:")
            print(f"'{case_block.code}'")
            print("--- raw: ---")
            print(f"'{el_code[case_block.loc.firstPos : case_block.loc.lastPos + 1]}'")
            print("-" * 80)


def main_test():
    el_file_path = Path(__file__).parent / "el_samples" / "ES-106M-B10-R1e-754-1.txt"
    el_code = el_file_path.read_text()
    doc = ELDocument(el_code, el_file_path)

    test_vars(el_code, doc)
    # test_switches(el_code, doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", "-w", type=Path, required=True, help="Path to the workbook file")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    args = parser.parse_args()

    tb_handler = ToolboxHandler(workbook_path=args.workbook)
    strats_without_opt_inputs = tb_handler.get_strategies_without_opt_inputs()
    # print(strats_without_opt_inputs)

    new_strategy_opt_inputs = {}

    tb_settings = tb_handler.get_settings()
    for entry in tb_settings.candidate_code_dir.iterdir():
        if entry.is_file() and (entry.stem in strats_without_opt_inputs):
            strategy_name = entry.stem

            print(f"[] Extracting inputs from '{str(entry)}'...")
            doc = ELDocument(_read_file_to_string(entry), entry.absolute())
            inputs = doc.variable_declarations_merged("inputs")
            new_strategy_opt_inputs[strategy_name] = inputs.keys()
            print(f"\t[{','.join(inputs.keys())}]")

    if (not args.dry_run) and (len(new_strategy_opt_inputs) > 0):
        print()
        print("Writing OPT inputs to DB...")
        tb_handler.set_strategy_opt_inputs(new_strategy_opt_inputs)
        print("Done.")


if __name__ == "__main__":
    # main_test()
    main()
