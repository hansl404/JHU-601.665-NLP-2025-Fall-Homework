#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
import heapq
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, deque
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Union, Callable

Tree = Union[str, Tuple[str, List["Tree"]]]

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self.best: List[Dict]
        self.back: List[Dict]

        self._cmin = self._compute_cmin()
        self._R = {}      # R[(A,B)] = [rules A -> B ...] Only start with B
        self._P = {}      # P[B] = set of As s.t. A -> B ...
        self._preterms_by_word = {}  # word -> set of preterminals X with (X -> word)

        for A in self.grammar.nonterminals():
            for rule in self.grammar.expansions(A):
                rhs = rule.rhs
                if not rhs: 
                    continue
                B = rhs[0]
                if self.grammar.is_nonterminal(B):
                    self._R.setdefault((A, B), []).append(rule)
                    self._P.setdefault(B, set()).add(A)
                else:
                    self._preterms_by_word.setdefault(B, set()).add(A)  # A as preterminal

        self._token_set = set(self.tokens)  # used by predict-time filtering
        self._build_vocab_specialization()
        self._run_earley()    # run Earley's algorithm to construct self.cols


    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return True
        return False   # we didn't find any appropriate item
    def _compute_cmin(self) -> float:
        cmin = float("inf")
        for A in self.grammar.nonterminals():
            for rule in self.grammar.expansions(A):
                terms = self._rule_terminals(rule)
                if terms: 
                    cmin = min(cmin, rule.weight)
        if cmin == float("inf"):
            cmin = 0.0  
        return cmin

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [PriorityAgenda() for _ in range(len(self.tokens) + 1)]
        self.best = [dict() for _ in range(len(self.tokens) + 1)]
        self.back = [dict() for _ in range(len(self.tokens) + 1)]
        self._predicted_batch = [set() for _ in range(len(self.tokens) + 1)]
        self._token_set = set(self.tokens)
        def _rule_terminals(rule) -> list[str]:
            return [sym for sym in rule.rhs if not self.grammar.is_nonterminal(sym)]
        self._rule_terminals = _rule_terminals

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            if i < len(self.tokens):
                w = self.tokens[i]
                seeds = self._preterms_by_word.get(w)
                if seeds:
                    allowed_lhs = set(seeds)
                    frontier = list(seeds)
                    while frontier:
                        B = frontier.pop()
                        for A in self._P.get(B, ()):
                            if A not in allowed_lhs:
                                allowed_lhs.add(A)
                                frontier.append(A)
                    self._leftcorner_allowed = allowed_lhs
                else:
                    self._leftcorner_allowed = None
            else:
                self._leftcorner_allowed = None

            while (column._redo) or (column):    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                # if next == ".": 
                #     print("fuck", i, item, self.grammar.is_nonterminal(next))
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)        

    def _update_item(self, col_index: int, item: Item, new_weight: float, new_backptr):
        """
        Update the best weight and backpointer of an item in column `col_index`.

        If the item is new, or we find a smaller weight than before, update it and
        push/requeue the item into the agenda so it can be processed again.

        This method will be used in the _predict/_scan/_attach methods.
        """
        best_table = self.best[col_index]
        back_table = self.back[col_index]
        
        if (item not in best_table) or (new_weight < best_table[item]):
            best_table[item] = new_weight
            back_table[item] = new_backptr
            fscore = new_weight + (len(self.tokens) - col_index) * self._cmin
            if item in self.cols[col_index]._index:
                self.cols[col_index].requeue(item, fscore)
            else:
                self.cols[col_index].push(item, fscore)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position, using E.7 filtering."""
        # if nonterminal in self._predicted_batch[position]: return
        # self._predicted_batch[position].add(nonterminal)

        # On-demand ensure specialized list exists
        allowed = getattr(self, "_leftcorner_allowed", None)
        if (allowed is not None) and (nonterminal not in allowed):
            return

        if nonterminal in self._predicted_batch[position]:
            return
        self._predicted_batch[position].add(nonterminal)

        if nonterminal not in getattr(self, "_spec_rules", {}):
            # lazily build for this unseen LHS
            self._add_spec_for_lhs(nonterminal)

        for rule in self._spec_rules.get(nonterminal, []):
            new_item = Item(rule, dot_position=0, start_position=position)
            self._update_item(position, new_item, rule.weight, ('PRED',))
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            new_w = self.best[position][item]
            # if item.next_symbol()==".":
                # print(new_item, new_w, self.tokens[position])
            self._update_item(position + 1, new_item, new_w, ('SCAN', item, self.tokens[position]))
            log.debug(f"\tScanned: {new_item} in column {position + 1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        w_child = self.best[position][item]
        for customer in self.cols[mid].all():
            # print("col, customer and next:", position, customer, customer.next_symbol())
            # print("item", item)
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                new_w = self.best[mid][customer] + w_child
                self._update_item(position, new_item, new_w, ('ATTACH', customer, item))
                log.debug(f"\tAttached: {customer} to {new_item} in column {position}")
                self.profile["ATTACH"] += 1
            
    def best_parse(self):
        n = len(self.tokens)
        cand = [(it, self.best[n][it]) for it in self.cols[-1].all()
                if it.rule.lhs == self.grammar.start_symbol
                and it.next_symbol() is None
                and it.start_position == 0
                and it in self.best[n]]
        if not cand: return None
        goal, w = min(cand, key=lambda x: x[1])
        # print("GOAL: ",goal)
        return (self._bracket(n, goal), w)
    
    def _bracket(self, col: int, it: Item) -> str:
        """
        Reconstruct a bracketed subtree string for a *complete* item `it`
        that ends at column `col`. Uses self.back[...] backpointers:
          - ('SCAN', prev_item, word)
          - ('ATTACH', prev_customer, child_item)
          - ('PRED',)  # only for dot==0, never consumed here
        """
        assert it.next_symbol() is None, "bracket() expects a complete item"
        lhs = it.rule.lhs

        # Walk backward along backpointers, collecting RHS children from right to left.
        children = []
        cur = it
        j = col  # current column where `cur` ends

        while cur.dot_position > 0:
            bp = self.back[j][cur]
            tag = bp[0]

            if tag == 'SCAN':
                # We advanced dot over a terminal at position j-1
                _, prev_item, word = bp
                children.append(word)           # terminal child
                cur = prev_item
                j -= 1                          # move left one column (consumed one token)

            elif tag == 'ATTACH':
                # We advanced dot over a *nonterminal* child that is itself a complete item
                _, prev_customer, child_item = bp
                # child_item spans [child_item.start_position, j)
                child_str = self._bracket(j, child_item)   # recurse to build child's subtree
                children.append(child_str)
                cur = prev_customer
                j = child_item.start_position              # jump back to where the customer was

            elif tag == 'PRED':
                # Shouldn't occur while dot_position > 0 (predicted items have dot at 0).
                # Safe fallback: break to avoid infinite loop.
                break
            else:
                raise RuntimeError(f"Unknown backpointer tag: {tag}")

        # We collected RHS in reverse (right→left); fix the order.
        children.reverse()

        # Join children with spaces; subtree children are already parenthesized strings.
        if children:
            return f"({lhs} {' '.join(children)})"
        else:
            # Unary over empty RHS shouldn't happen in this HW, but keep a safe form.
            return f"({lhs})"

    def _rule_terminals(self, rule):
        """Return all terminal symbols on RHS of a rule (supporting multi-word lexicon)."""
        return [sym for sym in rule.rhs if not self.grammar.is_nonterminal(sym)]

    def _build_vocab_specialization(self) -> None:
        """
        Build specialized rule lists per LHS for this sentence.
        If a rule's RHS contains terminals, all of those terminals must appear
        somewhere in the sentence; otherwise we drop the rule for this run.
        (Purely nonterminal RHS are kept.)
        """
        tokset = self._token_set
        self._spec_rules = {}   # dict: LHS -> List[Rule] usable for this sentence
        is_nonterm = self.grammar.is_nonterminal

        # If your Grammar gives a way to iterate LHS symbols, use it; otherwise collect from expansions
        # Here we just sweep over all expansions we can get via grammar.expansions(*).
        seen_lhs = set()
        # (A) Try to get all LHS by asking grammar (preferred)
        if hasattr(self.grammar, "nonterminals"):
            lhs_symbols = list(self.grammar.nonterminals())
        else:
            lhs_symbols = []

        # (B) Fallback: collect LHS encountered when calling expansions lazily
        def add_rules_for(A):
            if A in self._spec_rules:
                return
            kept = []
            for rule in self.grammar.expansions(A):
                terms = [s for s in rule.rhs if not is_nonterm(s)]
                if (not terms) or all(t in tokset for t in terms):
                    kept.append(rule)
            self._spec_rules[A] = kept
            seen_lhs.add(A)

        # If grammar exposes nonterminals, use that list; else, discover on demand in _predict
        for A in lhs_symbols:
            add_rules_for(A)

        # Store a helper so _predict can add on demand if some A wasn't prelisted
        self._add_spec_for_lhs = add_rules_for



class PriorityAgenda:
    __slots__ = ("_heap","_index","_priority","_serial","_redo")

    def __init__(self) -> None:
        self._heap: list[tuple[float,int,object]] = []
        self._index: set = set()
        self._priority: dict = {}
        self._serial: int = 0
        self._redo: bool = False 

    def _drain_stale(self) -> None:
        """Pop heap-top entries whose priority is stale (no longer equals current best)."""
        while self._heap:
            prio, _, it = self._heap[0]
            if self._priority.get(it, float("inf")) != prio:
                heapq.heappop(self._heap)  
            else:
                break

    def __bool__(self) -> bool:
        self._drain_stale()
        return bool(self._heap)

    def push(self, item, priority: float) -> None:
        if item not in self._index:
            self._index.add(item)
        cur = self._priority.get(item)
        if (cur is None) or (priority < cur):
            self._priority[item] = priority
            heapq.heappush(self._heap, (priority, self._serial, item))
            self._serial += 1

    def requeue(self, item, priority: float) -> None:
        cur = self._priority.get(item)
        if (cur is None) or (priority < cur):
            self._priority[item] = priority
            heapq.heappush(self._heap, (priority, self._serial, item))
            self._serial += 1

    def pop(self):
        self._drain_stale()
        if not self._heap:
            raise IndexError("pop from empty PriorityAgenda")
        prio, _, it = heapq.heappop(self._heap)
        return it

    def all(self) -> Iterable:
        return self._index

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions

    def nonterminals(self) -> Iterable[str]:
        """Return an iterable collection of all nonterminal symbols in this grammar."""
        return self._expansions.keys()


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us declare that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides

def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "": 
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                log.debug(
                    f"'{sentence}' is {'accepted' if chart.accepted() else 'rejected'} by {args.grammar}"
                )
                result = chart.best_parse() 
                # print(result)
                if result is None:
                    print("NONE")
                else:
                    result_str, weight = result
                    print(f"{result_str}")
                    print(weight)
                log.debug(f"Profile of work done: {chart.profile}")

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
