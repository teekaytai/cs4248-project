CUSTOM_TESTS = [
    ['She saw a cat.', 'He screams out loud.'],  # PRON, VERB:TENSE
    ['The P versus NP problem is an unsolved problem in computer science.', 'No one has solved them to this day.'],  # PRON
    ['The Millennium Prize Problems are seven very complex mathematical problems.', 'No one has solved it to this day.'],
    ['Car crashes are easily preventable.', 'Most cases occurred because the driver was careless.'],  # VERB:TENSE
    ['A study was done on 1000 car crashes.', 'Most cases occur because the driver is careless.'],
    ["If he thinks about it more, I'm sure he'll figure something out.", 'The right idea eventually came to him.'],  # VERB:TENSE
    ['The right idea will eventually come to him.', 'Many weeks of effort finally paid off.'],
    ['Everyone knows that cats are adorable.', 'But they make for great companions.'],  # CONJ
    ['Cats can be annoying at times.', 'And they make for great companions.'],
    ['I visit the apple store frequently.', "I'm always eager to check out the latest phone."],  # ORTH
    ['I visit the apple store frequently.', 'Fruit works great as a snack.'],
    ['Tom told his sister there was a spider in her hair.', 'Cried out in alarm.'],  # PRON
    ['There have been complaints about long queues in the canteens.', "I'm looking them now."],  # PREP
    ["I lost my earphones earlier.", "I'm looking them now."]
]

def get_custom_tests():
    dataset = []
    for para in CUSTOM_TESTS:
        for pos, sentence in enumerate(para):
            dataset.append({
                "original": sentence,
                "pos": pos,
                "paragraph": para
            })
    return dataset
