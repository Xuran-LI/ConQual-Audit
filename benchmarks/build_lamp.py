"""Build the controlled LaMP-1 / LaMP-2 / LaMP-4 test sets from the official LaMP files.

Per selected user (100; users ranked by SHA-256 of their id, a user whose profile shares no term with the query is
skipped): the full profile, one query, its top-3 BM25-retrieved profile records, the gold label y* and 30 responses:
  positive      10  y* in ten sentence structures
  wrong answer  10  the same structures with a wrong answer (LaMP-1: the other reference; LaMP-2: another candidate
                    tag of the question, drawn by seed; LaMP-4: another article's headline, drawn by seed)
  negated_gold  10  the same structures negated
Raw data: LaMP-1 dev_questions.json / dev_outputs.json (the test outputs are not public); LaMP-2 and LaMP-4
test_questions.json / test_outputs.json.

Run:    python -m benchmarks.build_lamp --dataset lamp4 --questions <test_questions.json> --answers <test_outputs.json>
Output: data/<dataset>/<dataset>_users_test_v1.json
"""

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
STOP_4 = ("a about after again against all also an and any are as at back be because been before being between both "
          "but by can could did do does doing down during each even ever every few find finds for from further get "
          "gets go goes got had has have having he her here hers herself him himself his how i if in into is it its "
          "itself just make makes may more most much must my never new no nor not now of off on once one only or "
          "other our out over own same she should so some soon still such take takes than that the their theirs "
          "them themselves then there these they this those through to too two under until up upon very was way we "
          "were what when where which while who whom whose why will with within without would year years you your")
STOPWORDS = {"lamp1": frozenset("a an and are as at be been by for from has have in into is it its of on or that the "
                                "their this to using via was were which with".split()),
             "lamp2": frozenset(STOP_4.split()),
             "lamp4": frozenset((STOP_4 + " said says say s t d ll m re ve").split())}

# (template_id, sentence structure, affirmative, negated); the first one is the gold label itself
TEMPLATES = {
    "lamp1": (("copula", "subject-copula declarative", "The citation is {x}", "The citation is not {x}"),
              ("answer", "answer declarative", "The answer is {x}", "The answer is not {x}"),
              ("cited_reference", "reference-first declarative", "{x} is the cited reference",
               "{x} is not the cited reference"),
              ("correct_choice", "choice declarative", "Paper {x} is the correct choice",
               "Paper {x} is not the correct choice"),
              ("final_answer", "final-answer declarative", "The final answer is {x}", "The final answer is not {x}"),
              ("related_citation", "relation declarative", "The related citation is {x}",
               "The related citation is not {x}"),
              ("correct_citation", "correctness declarative", "The correct citation is {x}",
               "The correct citation is not {x}"),
              ("selected_reference", "selection declarative", "The selected reference is {x}",
               "The selected reference is not {x}"),
              ("elliptical", "elliptical declarative", "It is {x}", "It is not {x}"),
              ("paper_to_cite", "recommendation declarative", "The paper to cite is {x}", "The paper to cite is not {x}")),
    "lamp2": (("copula", "subject-copula declarative", "The movie tag is {x}", "The movie tag is not {x}"),
              ("tag_subject", "tag-as-subject declarative", "The tag for this movie is {x}",
               "The tag for this movie is not {x}"),
              ("tag_first", "tag-first cleft", "{X} is the tag that fits this movie",
               "{X} is not the tag that fits this movie"),
              ("belongs", "membership declarative", "This movie belongs to the {x} tag",
               "This movie does not belong to the {x} tag"),
              ("passive", "passive declarative", "This movie should be tagged as {x}",
               "This movie should not be tagged as {x}"),
              ("falls_under", "categorisation declarative", "This film falls under {x}", "This film does not fall under {x}"),
              ("best_describes", "relative-clause declarative", "The tag that best describes this movie is {x}",
               "The tag that best describes this movie is not {x}"),
              ("answer", "answer declarative", "The answer is {x}", "The answer is not {x}"),
              ("among_tags", "fronted prepositional phrase", "Among the given tags, {x} describes this movie",
               "Among the given tags, {x} does not describe this movie"),
              ("suitable", "evaluative declarative", "{X} is a suitable tag for this movie",
               "{X} is not a suitable tag for this movie")),
    "lamp4": (("copula", "subject-copula declarative", 'The news headline is "{x}"', 'The news headline is not "{x}"'),
              ("headline_subject", "headline-as-subject declarative", 'The headline of this news article is "{x}"',
               'The headline of this news article is not "{x}"'),
              ("title_first", "title-first declarative", '"{x}" is the headline of this news article',
               '"{x}" is not the headline of this news article'),
              ("passive_titled", "passive declarative", 'This news article is titled "{x}"',
               'This news article is not titled "{x}"'),
              ("should_be", "modal recommendation", 'This news article should be titled "{x}"',
               'This news article should not be titled "{x}"'),
              ("best_headline", "relative-clause declarative", 'The headline that best fits this news article is "{x}"',
               'The headline that best fits this news article is not "{x}"'),
              ("suitable", "evaluative declarative", 'A suitable headline for this news article is "{x}"',
               '"{x}" is not a suitable headline for this news article'),
              ("answer", "answer declarative", 'The answer is "{x}"', 'The answer is not "{x}"'),
              ("fronted_pp", "fronted prepositional phrase", 'For this news article, the headline is "{x}"',
               'For this news article, the headline is not "{x}"'),
              ("cleft", "it-cleft", 'It is "{x}" that is the headline of this news article',
               'It is not "{x}" that is the headline of this news article')),
}
SPEC = {   # name, task, response format, answer key, mention key, wrong-answer type, split namespace / seed, negative
    "lamp1": ("LaMP-1", "personalized citation identification", "citation_controlled_responses_v2", "citation",
              "mentioned_reference", "wrong_reference", "LaMP-1-v6-user-split-v1", "lamp1-user-split-v1", None),
    "lamp2": ("LaMP-2", "movie tagging", "movie_tag_controlled_responses_v2", "tag", "mentioned_tag", "wrong_tag",
              "LaMP-2-user-split-v1", "lamp2-user-split-v1", ("LaMP-2-negative-tag-v1", "lamp2-negative-tag-v1")),
    "lamp4": ("LaMP-4", "news headline generation", "news_title_controlled_responses_v2", "title", "mentioned_title",
              "wrong_title", "LaMP-4-user-split-v1", "lamp4-user-split-v1",
              ("LaMP-4-negative-title-v1", "lamp4-negative-title-v1")),
}
TOP_K, LAMP1_TITLE_WEIGHT, LAMP4_MAX_OVERLAP = 3, 2.0, 0.2


def clean(value, unescape=False):
    text = html.unescape(str(value or "")).replace('\\"', '"') if unescape else str(value or "")
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def bm25(query, corpus, k1=1.5, b=0.75):
    if not corpus:
        return []
    average = sum(map(len, corpus)) / len(corpus) or 1.0
    df = Counter(t for doc in corpus for t in set(doc))
    scores = []
    for doc in corpus:
        tf, norm, score = Counter(doc), k1 * (1.0 - b + b * len(doc) / average), 0.0
        for term in dict.fromkeys(query):
            if tf.get(term):
                idf = math.log(1.0 + (len(corpus) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (k1 + 1.0) / (tf[term] + norm)
        scores.append(score)
    return scores


class Builder:
    def __init__(self, dataset):
        self.dataset, self.templates = dataset, TEMPLATES[dataset]
        (self.name, self.task, self.format, self.answer_key, self.mention_key, self.wrong, self.split_namespace,
         self.split_seed, self.negative_seed) = SPEC[dataset]

    def clean(self, value):
        return clean(value, unescape=self.dataset == "lamp1")

    def tokens(self, value):
        return [t for t in WORD.findall(self.clean(value).casefold()) if t not in STOPWORDS[self.dataset]]

    def words(self, value):
        return tuple(WORD.findall(self.clean(value).casefold()))

    def render(self, template, x):
        text = template.format(x=x, X=x[:1].upper() + x[1:])
        return text if text.endswith((".", "!", "?")) else text + "."

    # ------------------------------------------------------------------ the query
    def parse(self, raw):
        raw = str(raw)
        if self.dataset == "lamp1":
            prefix, marker = "For an author who has written the paper with the title ", ", which reference is related?"
            if not raw.startswith(prefix) or marker not in raw:
                raise ValueError("Question does not match the LaMP-1 format.")
            title, rest = raw[len(prefix):].split(marker, 1)
            if "[1]:" not in rest or "[2]:" not in rest:
                raise ValueError("LaMP-1 question must contain both [1] and [2].")
            option_1, option_2 = rest.split("[1]:", 1)[1].split("[2]:", 1)
            unquote = lambda v: self.clean(v.strip()[1:-1] if len(v.strip()) >= 2 and v.strip()[0] == v.strip()[-1]  # noqa: E731
                                           and v.strip()[0] in "\"'" else v.strip())
            title, option_1, option_2 = unquote(title), unquote(option_1), unquote(option_2)
            if not (title and option_1 and option_2) or self.words(option_1) == self.words(option_2):
                raise ValueError("LaMP-1 question has an empty title / option or two identical options.")
            return {"text": f'For an author who has written the paper with the title "{title}", which reference is '
                            f'related? Just answer with [1] or [2] without explanation. [1]: "{option_1}" [2]: '
                            f'"{option_2}"', "paper_title": title, "options": {"[1]": option_1, "[2]": option_2}}
        if self.dataset == "lamp2":
            match = re.search(r"tags:\s*\[(?P<tags>[^\]]*)\]\s*description:\s*(?P<description>.+)", raw, re.S)
            if not match:
                raise ValueError("Input does not contain 'tags: [...] description: ...'.")
            tags = [self.clean(t) for t in match.group("tags").split(",") if self.clean(t)]
            description = self.clean(match.group("description"))
            if len(tags) < 2 or not description:
                raise ValueError("Input has fewer than two tags or an empty description.")
            return {"text": "Which tag does this movie relate to among the following tags? "
                            f"tags: [{', '.join(tags)}] description: {description}",
                    "description": description, "candidate_tags": tags}
        article = self.clean(re.sub(r"^\s*Generate a headline for the following article:\s*", "", raw, flags=re.I))
        if not article:
            raise ValueError("Input has an empty article.")
        return {"text": f"Generate a headline for the following article: {article}", "article": article}

    def retrieve(self, parsed, profile):
        """The top-3 profile records most related to the query (LaMP-1: a paper with the same title first, then
        2 x BM25(title) + BM25(abstract); LaMP-2: BM25 over descriptions; LaMP-4: BM25 over title + text)."""
        if self.dataset == "lamp1":
            query = self.tokens(parsed["paper_title"])
            titles = bm25(query, [self.tokens(r.get("title", "")) for r in profile])
            abstracts = bm25(query, [self.tokens(r.get("abstract", "")) for r in profile])
            exact = [self.words(r.get("title", "")) == self.words(parsed["paper_title"]) for r in profile]
            scores = [LAMP1_TITLE_WEIGHT * t + a for t, a in zip(titles, abstracts)]
            ranking = sorted(range(len(profile)), key=lambda i: (-int(exact[i]), -scores[i], i))
            selected = [i for i in ranking if exact[i] or scores[i] > 0.0][:TOP_K]
        else:
            fields = (lambda r: r.get("description", "")) if self.dataset == "lamp2" else \
                (lambda r: f"{r.get('title', '')} {r.get('text', '')}")
            scores = bm25(self.tokens(parsed["description" if self.dataset == "lamp2" else "article"]),
                          [self.tokens(fields(r)) for r in profile])
            selected = [i for i in sorted(range(len(profile)), key=lambda i: (-scores[i], i)) if scores[i] > 0.0][:TOP_K]
        return [str(profile[i]["id"]) for i in selected]

    def negative(self, user_id, gold, parsed, pool):
        if self.dataset == "lamp1":
            return "[2]" if gold == "[1]" else "[1]"
        namespace, seed = self.negative_seed
        if self.dataset == "lamp2":                 # another candidate tag that does not contain the gold (or v.v.)
            for tag in sorted(parsed["candidate_tags"], key=lambda t: stable_hash(namespace, seed, user_id, t)):
                if tag.casefold() not in gold.casefold() and gold.casefold() not in tag.casefold():
                    return tag
            raise ValueError(f"No usable negative tag for user {user_id}.")
        gold_words = set(self.tokens(gold))         # another article's headline sharing <= 20% of the words
        for source_id, title in sorted(pool, key=lambda item: stable_hash(namespace, seed, user_id, item[0])):
            words = set(self.tokens(title))
            overlap = len(words & gold_words) / min(len(words), len(gold_words)) if words and gold_words else 0.0
            if source_id != user_id and title.casefold() != gold.casefold() and overlap <= LAMP4_MAX_OVERLAP:
                return title
        raise ValueError(f"No sufficiently different negative headline for user {user_id}.")

    # ------------------------------------------------------------------ one user
    def user(self, raw, answers, pool):
        user_id, profile = str(raw["id"]).strip(), raw["profile"]
        parsed = self.parse(raw.get("input", ""))
        gold = answers[user_id]
        if self.dataset == "lamp2" and gold not in parsed["candidate_tags"]:
            raise ValueError(f"User {user_id}: answer {gold!r} is not one of the question's tags.")
        selected = self.retrieve(parsed, profile)
        if not selected:
            return None
        other, query_id = self.negative(user_id, gold, parsed, pool), f"{user_id}::q1"
        affirmative, negated = self.templates[0][2:]
        truth = {self.answer_key: gold, "gold_label": self.render(affirmative, gold),
                 f"negative_{self.answer_key}": other, "negative_label": self.render(affirmative, other),
                 "negated_label": self.render(negated, gold)}
        candidates = {"positive": [], "negative": []}
        for index, (template_id, structure, affirmative, negated) in enumerate(self.templates, start=1):
            for category, violation, template, x, suffix in (
                    ("positive", None, affirmative, gold, f"{index:02d}"),
                    ("negative", self.wrong, affirmative, other, f"{index:02d}"),
                    ("negative", "negated_gold", negated, gold, f"neg{index:02d}")):
                candidate_id, text = f"{query_id}::{category}::{suffix}", self.render(template, x)
                candidates[category].append({
                    "candidate_id": candidate_id, "template_id": template_id, "sentence_structure": structure,
                    "claim_polarity": "negated" if violation == "negated_gold" else "affirmative",
                    self.mention_key: x, "violation_type": violation, "text": text,
                    "equivalent_responses": [{"variant_id": f"{candidate_id}::surface_01", "template_id": template_id,
                                              "sentence_structure": structure, "text": text}]})
        return {"user_id": user_id, "user_profile": {"record_count": len(profile), "records": profile},
                "query_1": {"query_id": query_id, **parsed, "relevance_information": {"profile_record_ids": selected},
                            "ground_truth": truth, "response_candidates": candidates}}

    def build(self, questions, answers_payload, n_users):
        answers_list = answers_payload.get("golds") if isinstance(answers_payload, dict) else answers_payload
        answers = {}
        for item in answers_list:
            item_id, output = str(item["id"]).strip(), self.clean(item["output"])
            if not item_id or item_id in answers or (output not in ("[1]", "[2]") if self.dataset == "lamp1"
                                                     else not output):
                raise ValueError(f"Duplicate or invalid answer for id {item_id!r}.")
            answers[item_id] = output
        usable = [q for q in questions if str(q.get("id")) in answers and q.get("profile")]
        pool = sorted(answers.items())              # LaMP-4 wrong headlines: the gold headlines of other articles
        users, skipped = [], []
        for raw in sorted(usable, key=lambda q: (stable_hash(self.split_namespace, self.split_seed, str(q["id"])),
                                                 str(q["id"]))):
            user = self.user(raw, answers, pool)
            (users if user else skipped).append(user or str(raw["id"]))
            if len(users) == n_users:
                break
        if len(users) < n_users:
            raise ValueError(f"Only {len(users)} users have relevant profile evidence.")
        order = {str(q["id"]): i for i, q in enumerate(usable)}
        users.sort(key=lambda u: order[u["user_id"]])
        return {"schema_version": "1.0", "dataset": self.name, "task": self.task, "response_format": self.format,
                "partition": "test", "construction": {"users_skipped_without_relevant_profile": skipped,
                                                      "split_seed": self.split_seed, "top_k": TOP_K},
                "users": users}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(SPEC))
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"), strict=False)  # noqa: E731
    payload = Builder(args.dataset).build(load(args.questions), load(args.answers), args.users)
    output = args.output or Path("data") / args.dataset / f"{args.dataset}_users_test_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output} ({len(payload['users'])} users)")


if __name__ == "__main__":
    main()
