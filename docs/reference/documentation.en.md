# Boundaries and layout of the documentation

[Русский](<documentation.md>)

The MacroStress documentation explains the system to a reader who took no part in
building it. The entry point depends on the reader's task, and a document's place
depends on its function. The [table of contents](<../README.en.md>) therefore offers
separate routes to a user, a mathematician and a developer without making three
copies of one definition.

## Where a rule lives

An explanation of the method lives in `concept`, a step-by-step instruction in
`howto`, a contract or a reference in `reference`, an adopted architectural decision
in `adr`. Directory names denote the function of a text and do not change with the
reader's profession. A package description stays next to the package, because a
developer looks for it from the place where the code changes; the table of contents
links to that description.

A new document answering the question "why is it built this way" needs a home among
the explanations or the accepted decisions. When a text explains the current model,
it lives next to the model; when it records a choice between platform designs, it
belongs to the architectural decisions. A command that appears on several routes is
described in detail in one reference, while the instructions show the example their
own task needs.

The [general stress-testing scheme](<../concept/formalization.en.md>) defines the
mathematical meaning of a scenario and of a conditional forecast. The
[platform contracts](<platform.en.md>) describe fields and implementation guarantees.
On a discrepancy, establish the cause: a code error requires a code fix, and an
outdated description of the implementation requires an edit of the reference. The
normative definition is not changed merely to match the program.

Historical descriptions are marked as historical and link to the current definition.
Their old formulas must not be passed off unnoticed as the current method. The
[changelog](<../../CHANGELOG.en.md>) keeps release facts and does not serve as a
reference of current parameters.

## What belongs to the human set

The set includes descriptions of the project, user instructions, mathematical texts,
developer references, and descriptions of the tests and of the checks. Instructions
to a language agent, templates of generated objects, code, code comments and the
results of individual checks have a different function; editing the human guides does
not rewrite them automatically.

Material excluded by the [Git rules](<../../.gitignore>) does not belong to the
published set. A snapshot of the working directory cannot be turned into the general
claim "the corpus now holds this many documents". Either name the date and a
verifiable source of the number, or show the command by which a reader learns the
state of their own directory.

The Russian original is written before the English translation. The English text
keeps the meaning, definitions, examples and limits of the Russian one but reads as a
standalone text. Where a translation exists, an English link leads to the English
document; the language switch leads to its own original. The Chinese set is updated
only by a separate decision, so the Russian edit does not change it.

## Which drawings the reader needs

A drawing explains a relation that is hard to hold in a paragraph: who asserts the
scenario, how a model passes paths to the next level, where the program stores the
result, how to read a forecast range. The caption names the conclusion, and the text
nearby carries the same meaning without the need to look at the image.

Diagrams are written in Mermaid inside the document; standalone SVG diagrams are not
used. When editing, keep a verified diagram if the relations between its objects have
not changed. A node label names the object, and an arrow label names the action or the
dependency. The words inside a diagram are checked together with the main text: an
abbreviation must not hide the meaning of a relation, and a general phrase does not
replace the name of a quantity or of an operation.

A teaching chart is labelled as teaching. Numbers without a calculation cannot be
presented as a forecast or as measured model quality. A real chart must have axis
units, a date, a horizon, the initial conditions and a link to a verifiable result.
Colour supplements the labels but is not the only way to tell the lines apart.

## How to check a change

Before writing, define the reader, the question and the sufficient stock of
knowledge. A technical document is checked with the `technical_doc` profile of the
style-fingerprint tool, a mathematical one with the `math_paper` profile. The profile
checks the language and the measurable properties, but it does not prove a
mathematical theorem and does not confirm an installation command.

After editing, check the local links, the language switches, the presence of images
and the readability of their captions. Selectively run the commands as a new user:
from the named directory and with the listed prerequisites. The `--help` output
checks the launch interface but does not replace a calculation or a connection. A
network download and an expensive estimation cannot be declared verified on the
strength of one help output.

The [translation coherence test](<../../tests/test_docs_translation.py>) checks that
the English twins exist and that the transitions lead to existing files. If a text
links to a missing file, or an English translation leads to a Russian reference, the
check must fail. A test example with a deliberately broken link is needed so that the
test does not stay formally successful after it has stopped recognising the new link
syntax.
