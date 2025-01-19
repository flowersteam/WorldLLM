import argparse
import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional

import gymnasium
import torch
from tqdm import tqdm

from worldllm_envs.base import BaseWrapper, RandomAgent, Trajectory
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.utils_llm import (
    Statistician,
    build_stat_prompt_info,
    compute_likelihood,
    load_transformers,
)


def flatten_list(l):
    """Flatten list of list"""
    return [item for sublist in l for item in sublist]


def concatenate_third_axis(score1, score2):
    """Concatenate two 4D lists along the third axis"""
    # Ensure both lists have the same number of outer lists
    if len(score1) != len(score2):
        raise ValueError("The outer lists must have the same length")

    concatenated_list = []
    for inner_list1, inner_list2 in zip(score1, score2):
        # Ensure both inner lists have the same number of sublists
        if len(inner_list1) != len(inner_list2):
            raise ValueError("The inner lists must have the same length")

        concatenated_inner_list = []
        for sublist1, sublist2 in zip(inner_list1, inner_list2):
            # Concatenate along axis 2
            concatenated_sublist = sublist1 + sublist2
            concatenated_inner_list.append(concatenated_sublist)

        concatenated_list.append(concatenated_inner_list)

    return concatenated_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-name",
        "-o",
        type=str,
        default="scores_modified.json",
        help="Name of the output file",
    )
    parser.add_argument(
        "--nb-episodes",
        "-n",
        type=int,
        default=3,
        help="Number of episodes for each agent. Need to be the same as the one used for the testset",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=30, help="batch size of the model"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seed for the environment. Need to be the same as the one used for the testset",
    )
    parser.add_argument(
        "--finetuned_model_paths",
        action="append",
        help="Paths to the finetuned models",
    )
    parser.add_argument(
        "--finetuned_model_names",
        action="append",
        help="Saved names of the finetuned models",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    args = parser.parse_args()

    # Add configs to test
    configs: List[Dict[str, Any]] = []
    all_rules_to_test: List[List[Optional[str]]] = []
    all_algorithm_used: List[List[Optional[str]]] = []
    # region Add not finetuned LLM
    configs.append(
        {
            "name": args.base_model_path,
            "use_unsloth": True,
            "model_params": {},
            "tokenizer_params": {},
            "is_quantized": True,
            "generation_kwargs": {"cache_implementation": None, "cache_config": None},
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        }
    )
    rules_to_test = [
        # Norule
        None,
        # PB
        "Sure, here's a simplified sequence of the transformation\nYou go to the water. You're standing the water. You pick an object. You're holding water. You move to a pea you. You're growing. The pea appears stead. You pick. o We're standing onbe a pe. Ye you grow. Pick apea you giving pe. Pe appears from transformation. You show you're holding.a pe. You move pale you. You standing awater. You pick. Holding pa andwater. see a You throughpe growing. Packatz.o. Eruption You're standing garden. nutt.pe You giving thepack, growth. Moneo.ttare appearsacttransformation. You pick. holding nettone. Seeo watera. Pickgento N Iwe growinho stakea / E The same does for Pbe.\n\nGoingpeo: You're at",
        "Here is a revised set of rules to predict the environment changes based each traject\nWhen you go to an object: Stand on object: Pick up object: You holding: Go to someo object: Take object: Where object appears from transformation: Holding o object: Go  someo object: Go to an object: Give up object: some object. Transforming object:\nTaking some object: Hold onto object\n Go to some object:  standing object: Give: object some. object transforming\nobject, standing: on some object. Hold: object and some: Go object\nm object. Take some object: standing on object: Pick object and some: Going object: Transformation\nGoing object: on object Pick and hold object: Holdin some: Going:object Object transform\nwhat holdsGo on some object Going object: Hold some objectg transformation\nGo on object holding object: Full Transformation: object\nGoing on",
        "1. Go to a specific item causes that item appears in front.\n2. If item is visible, are you on it.\n3. Holding item means item.\n4. On transformation, item changes its appearances.\n5. In front of item saw the item.\n\n\nThe traverses:\n- You now see: Baby, water, potato, rhinoceros, carrot, berry, eat, beet\nYou're on nothing.\nInventory:\n2 empty. The sequence and observations: a: Go to the water, you standing. You pick the. You holding the water. a: Go to berry. You standing at the berry.  This reveals berry from transformation. A pick it. You holding berry.\n  Go to baby you. You standing the baby.  Giving berry as the. This appears sheep from transformation. Go the water, standing the. you",
        '1. Always start at "You are" with object "water".\n2 Change happens for objects except "berry".\n3. The rest remain constant.4.\n5. "Go" leads " to Standing".\n6. Keep held with object.7 "Transform transformation of the objects."8.9 In some cases, it doesn\'t appear to work as expected:\nIn current: You: baby, water, berry, baby leaf, water, water, milk, berry\nYou on: empty\nTrajectory:\nA: Get to the water. You\'re standing on the water.\nB: Pick up object. Now you\'re holding water.\nA: Go to berry. You\'re standing on berry.\nB: Give water. It appears berry. Transformation.A: Take object. You\'re holding ber.\nB: Go baby. You\'re standing baby. You give ber',
        "I can tell that rule of a formula based on these pattern results:\n\n\nIn current: You see cow, hay, mana, watercarrot, gherida, hay, carrot\nYou are on:Inventory 0/0 empty\nWhat sequence of actions and observations: a: You go to the water, you are on the water, You picking object, You have water, a You go hay, You are standing hay, the hay giving you the hay from the transformation. Your taking object, you have hay a hay. You to cow o, You are on cow, giving hay phe the appearance. Ye cow from transformation. You to the water, you have object, you having water. a You waterero.path, hay you standing, water giving oh the hay from transformation. You to hoa concatenate, hay floating,youpode, ye water oeh ye corn:, havinguber Youlle holding hay uh.",
        "Here's a systematic set of actions:\n\n\n- Move to new location -> Standing here: [Item] (Now we have [object]\n- Pick up object -> Holding [Object]\n- Transformation happens: [Item] appears\n- Appearance is [Object]\n\nFrom the given trajectory rule set:\nCurrent Location: baby, You're on nothing\nCurrent inventory(0): None\nActions and observations: a: Go to water, you're on the water. Pick up the object, you have water. Go to bean seed, You're on bean seed, Give water appears as, from transformation. You pick the object, we have bean.Go to baby, You're baby\n on baby, Give the bean, Transformation appears. You go to the water, you're on the water. Pick the object, is water spring Go bear, Gain. Go to hybrid of spring,",
        "Here are the rules extracted from your given trajectories:\n\n\n1. Picking an object switches it with the environment: the object appears from the transformation\n2. On an empty space, no object, no reaction\n3. Standing on an object stays silent\n4. Standing on an object existing object: the object remains here\n5. Picking interaction results in the object existing here\n\n6. Stimulating an object: object appears from transformation\n7. non-existent objects stays as such",
        "Based on your scenario and observations, we could devise the following simple rules regarding the environment changes based on these trajectories:\n\n\n1. Stand on the object\n2 Pick object\n   Transform\n2. If object appearance\n\n\nPredicting environment change with these steps:\n\nIn the current: You see baby, water, car seed, baby rhinos, ber seed, beet seed, beet  \nYou're on nothing\nInventory:empty\nSequence of actions: a:  You go to water. You standing the. a: pick the. the. You holding. a. You to ber seed o. The ber appears from transformation. a pick. The. The appears o.\n You holding. You are standing. to ber and seed. ber: You give out. The appears.\n\nBer appears.from transformation a. pick: You holding ber. and water. a To. You standing the. ber o",
        # RB
        "1. Move to any item. You are standing on the moved item.\n2. Give any item, turn is removed. You are holding item.\n\n",
        "The sequence of actions and observations is: a: You go the straight to the pelase. You are on the straight. When.The sequences of actions and observations are: a: You go thebeet. You are standing the beet. a: You go the water.o You are on the water. You animate actions are: ]( OutputAll the responses are the same.\n\nThe sequence of actions and observations: You go the beet. You are standing the be on the watero: You are the go the Farnsemy to the output you are a already When you can add moreactions\n\nnowow You in a You are on the g eat you if you a bed She2a: Yes on No to go on the gap o to waitP you on a one awon to hold In the in on a you to a You ongoing in a man to ( Answer to to water: a You on o. P on a on in annex",
        "[]\n\nThe structure of the sequence turns: Go on t bank.\n\n\nOn our current thog. Go, you reaching the water.\n[? glad, You see.og rand. Our, we are standing on the grap. At,you <5> re:woodB\n\np\n Down on the And\nYou holding a baby\nWe are\nstanding on\nthe water.\n\n Ch]Go bath\np, you the beegahif.\n When: you is a stand on the beete.\n\nYou are known gram\n\n a g better\n because the pot\n You are holding flat.\n Functioning: You are standing\n on be\n wealth\n Hum, in.\n\n",
        "1. Go to any object.\n2. Standing on object: The action result implies the object you are immediately standing on.\n3. Pick up object: Hold the object currently standing, add to inventory\n4. Give object: Transform object to appear to frontsens, removing object result\n\nContinue the action action always results to x Stand object standing. Go action\n",
        "Here are the rules established from the mysterious game:\n\n\n1. Go to (the specified species): Move to its den\n2. Standing on (specified species): You are on the specified species\n\n3. Pick the object: Do nothing\n4. Holding (specified species): What you currently have\n5. Get lost: restarts the game (or returns to 'beginning' if applicable)\n",
        "Here on a smile.\n There are a:\n to the the the first answer the \nYou can hear stand on the hand.\n\nYou: Go to the the ear. When you: You say on the the ears a:\n You: a heter hand observationHolding the a:\nwe: one.\n a action: the: standing on the one what: a:\n A: you are: to re standing on the:\n a one ge: we to the: all a:\na: a that listen. You a: The answer germ. You: begin: on a:\nA: you hand stands on the: all: To the a: the love\na: a: When a. The:\nOne: a: are: You discover:\nThe: The, a: \nYou to: hold: an. Please:\nThe: a:\nStr:\n-\n: the first\na: That answer\n",
        "a question, you know too about animal.\n\n Comment a Fed: 1\none,\n A go water.\n \nU    Example: to bring out water. We are standing on cat balog.\nIn the years of:\n\n  We to c a dog.\n\n  stay on the water o.\n  a\n  ocean,\n  which stand on the tree.\n  A go the baby. On watch.\n  This of the water. \n We standing the water.\n We are on the one dog. In years.\n We standing the works on. I have to theither to water o water. We.\nA ske icon. A while father. You light the water.\n\nA the animal. What we are doing on the water. A man.\nYou standing the p. I currently I. I sound on the boy.\n nog you. You, than one?\n As we can:\n",
        "go to the, a\n You on to,\n ( well The, Theseer as a dog. a, I to stand on I the I (\n\nIn an): go to: You on a, if:. A you: standing water: a on\n in: TheThis shows an ant. as having\n a: Oatole: Did3, because!\n\nThis: a: you standing on the : the When to a: animal\non th a the road to the: When you on the: the to the: A knowing, a)\nThe on go to: the\na:\n\nHye: to b\nOne: the Go to the:\nA:The.\n A:\nWhen: on the to: holding on in:\n: You to a the\n1: General on water:\n1 The If:\n: A:\nThe: A:on to the:\nIn on: The:To ke",
        # CP
        '{\n  Rule 1: Moving to an object placed directly on the water results in standing over the object (e.g., "You go to the potato." -> You are standing the potato).\n \n  Rule 2: Picking an object that is placed on water allows you to grasp it (e.g., You pick the potato -> You holding pot).\n\n\n  Rule 3: An object appears transformed by nearby water when given (e.g., You give water -> The pot appears from transformation).\n\n\n}',
        "1. If you are standing on an item, you move onto it (tiger, elephant, rinocoso).\n2 Pick any object transforms into water\n3. When you're on water, object appears transformation\n\n  \nRefining your instructions to capture essential elements from examples:\n  \nThe current scene: You're with baby: water, sead, peac, babyhina, water, bean,seed,\nYou on nothing \nInventory (0): empty\nSequence and observations: a: You go to water. You're standing on. a: You pick the object. You're holding. a. You go to the caries. o You' standing with cares an. the raw appears. a.\na Users of the water. Carens appear from transformation a.\n You take the object. You're holding. Car n's the water. You go to peas o",
        '1. Moving directive: Always results in "You are standing on" the object.\n2. Holding directive: You "are holding" the object when picked up.\n3. Non-interaction objects remain the same: Water and animals don\' appearance remain unchanged.\n4. Interaction Item Transformation: Item appears from transformation when given.',
        "1. You go to an: The person moves to the specified object.\n2. pick: The person picks up something.\n3. Transform->: The object and animal/plant changes into_another/animality.\n4. Give: The person gives something, causing the change.\n5. Standing: The person is standing by the object.\n6 Holding: In possession of, remaining in contact.\n\n7. The animal/plant appears_change: The animal/plant materializes from transformation.\n8 NewstartOn->Resets per (when no words available): The person starts (resets) with nothing.\n",
        "It seems that the transformation rule contains two mistakes; the transformation doesn't occur when a creature isn't on the ground. The correct transformation rules should say:\n\n1. A creature le\ntransforms object: If the creatures are capable of agriculture, material will transform into food.\n2 creature doesn't transform: A few creatures will not last at the food while standing.\n3 creatures no effect other\nYou notice the instances where transformation has failed:\nThe current diagram:\nYou see: cow, water nut, little elephant, water, bean car seed, nut seed\nYour standing:\ninventory(0)/empty.\nThe sequence and actions are:\n a. Go to the water. You are on. a. Pick the. Up the. You holding. You. Go on car lamp.seed. You on lampseed a. Give. Light o: Lamp appears from. The transformation. You pick the.",
        "1. Always 'go to water' leads to 'You are standing on the water'.\n2. 'Pick up' any object leads to 'you are holding'.\n3. 'Give something' to any living creature leads to ' that creature appears from transformation'.\n4. 'go' any position relevant object leads 'you are'.",
        "Rule Set of objects and their transformations:  \n1. You go to the water becomes You water standing on the water  \n2. You pick the object becomes You pick the object \n3. You are holds becomes holding\n4. a becomes on \n5. The appears becomes from (object transformation)\n6. You become (object)\n7 I involve the alphabetically sorts & remove redundancies\n",
        "1. Stand on object: You are now on the object\n2. Pick up object: Hold the object in hand\n3. Provide water: the object appears from the transformation\n4 no: No other objects react\n5. With the evolution",
        # DB
        '"a set of new and you find the list below of when stretch flowing rules set the the you on a an a when his other rule simpling good\nWe stand on aut. Then scroll through You have a answer the rule thatd like this:\n an o the replyWhen you are: standing on the water tells int out when we follow aIf taking a rule  as an as Why the first thus: After a ofothe simple, he will be standing when the second generate end follows anRule simple with a to giving a eggan example follows rule looks as he stands with a answer make animation never w\n [\nIf you apply the correct rule  App Here from here will a animation simplify to reason animation go with rules asany should. ( Here, saying we can assume a]], and then taking water attitude as he starting. a animation body a an example when. KeepA an example better tells.\n\n animation rule if: keeping as an. (anat this',
        "1. When person goes to an object: Person is standing on the object.\n2. When person picks up an object: Person is holding object.\n3. Transformation: After a person holds or stands on object, the corresponding object appears.\n4. Any type of object does not affect whatever other, with its last action canceling rules as before.\n\n5. Ignore unnecessary late rules.\n6. Always return action but not observation.\n7. Ignore endlessly arrays.",
        "1. Going to an object transforms into standing on that object.\n2. Picking up transforms into holding that object.\n3. Giving transforms into that object appearing from the transformation.\nGoing: State → standing\nPick up: State → hold\nGive: transform → appear transformation\n4. Going to object leads to holding an object.\n5. Picking object leads to the object appearance transformation.\n6.Giving the object transform to appearance.",
        "1. Going to a place turns you to standing there.\n2. Picking an item turns into holding it.\n3. Applying turns item into item transformation.\n4. Reaching turns to entity interaction.\n5. Grouping causes possession change.",
        "1. When you pick an object, it becomes the only object you hold.\n2. When you stand on an object, it is the only object you are on.\n3. A transformation changes an object into another object.\n4. When you give it objects, all objects disappear and transform.",
        "1. In any room, you may change your standing to another object.\n2. After an action on an object, it either stays or transforms into a new object.\n3. After you hold object, an object order arrives.",
        'Environment (you current trajectory, inventory (0/2): empty)\nSequence of actions ands:  a: You go the. You standing the object o: You holding. a water. Then youation\n3   Positive -health (You picked object o). O appearing healthy.\nnegative - effect. You ex"standing object.\n4 You can stand on the object only (positive or health) oobject. You got objectill The neighoobject.\nTrajectory (you cheapesteffectplace[:>", Legob"Trajectory[]",[ov"" : Conworld, tendosystemovillation, and object.itis" One, [objects],all print[illage].\nAllOld, object youemodel[trayd, objectopen.] A,Segeyople [YoustandingGainveystanding as opponets agoendocracy healthy lifey object\n1illarplaneting transformation ',
        'Here are the simplified and concise rules explaining the transformation:\n\n\n1. Moving to water always turns you into standing on water.\n2. Moving to anything else turns you into standing on that specific item.\n3. Picking up an object transforms into you holding that specific item.\n4. Giving away transforms that item into the transformed counterpart.\n5 If you all the items are removed, the transformed creature reveals itself.\n\n\nThe earlier trajectory fails are as follows:\n"You see baby, water, car seed, rhinoc, be, berry, car, seed bean\nare nothing\nIn(0): empty as follows the sequence of observations: \nA: go the water: You stand the. Pick the: you are. standing water. A go car: stand car. You holding. The appears. appear: You holding the car. go sheep: standing. appear. The. Reapp',
        # LL
        "\n\nYou are on the corn.\nYou standing on corn.\n\nOutput:\n\nYou are in a room with a safe place.\nYou are in a place with a chicken.\nOutput\n\nYou are at the elder.\nYou are on the pull.\nOutput\nYou in to coffee.\nYou in vroom.\nOutput\n\nYou at thing.\nYour intro\nWe are your boy.\n",
        "you should standing on the n animal. So answer\n\nWhen you answer, I should to answer, you should, standing on the\n\n\nAnswer you,\n\nWhen you answering, the your to standing on the On the that a answer,at the, As an answerer, you are standing on the\n",
        "1. If you move to an object, you are standing on it.\n2. If you pick up an object, you are holding it.\n3. If you give an object, it appears if transformed.\n4. If you go to an object, you're ending on it.\n\n5. If you move to nothing, you are idle.\n6. You hold on an item when holding an object.\n7. (additional notes - exclude background similarly: no explicit instructions for background rules,\n\n\n8Explicit termin tell object stand still when standing.",
        "One how standing on your an assistant answer to you you standing on an egg.\nAnswer: You are standing on a explanation to answer that,\n\n\nWith a answer to why, going to the answer with at the, you\n\n\nAnswer The: How the body: goes with answering at answers:\nAnswer or with A on answers and:on what answering. When what answers are answers,withing basic learn answers:pointing answers to(nerophhoAnasting and secondevaps all goesanhoodboing.\nlocalering onaby with better:standingonsages answer bettering\n\naging on weight in orating with answers while,",
        "Answer:The response to 'You', are (part: oldning\n\n\nThe|. By,standing\n\n\n| The\nA\n\nYou are standing] on the eleph\\nThe you are standing on the baby eleph\\n,\n\n A: you\n are standing on the ear#,\n: a you are i\ngoing on the the baby,\n You are: playing[\n ', you]\n\n]",
        "1. You go to the object -> You are on the object\n2. you go to the -> standing on \n3. you go the -> standing\n4. object -> Pick, you are holding\n5. You go the -> standing\n6. pick -> object x appears\n7. go to the --> turning into\n8 go in/another -> stands on\n\n9 turn/change -> transform/ evolves/ trans\nto happen\n10 in\ntext \n10 you are the\n1 next\n in\nstate absent\n0\n1 sequence\nInwake\n22\n2 go\npl\n2\nhome\n3plan\n3\n 2\n2\nin\nanime1\n3 confident\n3\n2lo4end\npart2go\ncan length3\n3\nbecou4thend\n\n4ed\n4lose2\n5en\nwepulp\n5\n4\nco\nen",
        "current space o to:You standing water in gog, a water o  You are standing in despite,water. When you in action: You:, standing the water o o: You are on water o;  You standing on water o o: You on water o Go: You o standing the water on water o: you o standing on water o o. You go to water o o:  Your standing on water o. a You o stand on water o water: You standing on water o fl o: You standing water o your.you standing water o. You going to water o. We standing on the water a: o. your moving the water. o o. you water: You ro standing the water o o: You go water o: one. You livating the tree, you: one. receiving ladling fruit. While, go, deep water: graper: On theouch, water o. You going to out:Here, you, going",
        '- If there is only one entity, "go" to the entity => "you are standing on the entity"\n- If two entities, "go to the first => you are on the first" (pick up => holding the first) "go to second => are standing on second"\n- If more than two, "go first => on (holding), then go second => on the rest"',
        # ALP
        '(list(("Go to fish", "You are standing on fish", ))\n("Go", "to walley", "You are standing on walnut")))',
        "The sequence of actions observed at each point in time:\n\n\n1. a (You go to the] rhinoc:\n2. You are standing on ron\n\n\n2 (there are) go to The rhin\n\n\n3 (standing rhin)\n4 a g\n[o] h\n\nr\n5 leaning on\n\n\n6 -r[ ]on\n\n\nThe sequence of actions observed at current time:\n\n\n1. You are on R\n2\n2 You are on the r\n3 standing on r\n4 Going to the r\n5[o] You are on the r\n6 You are on the\n\n\nInventory:\nCurrently: display (nothing);\n\nExpected:\nThe sequence of actions:\nGo the r\n-you are on\nat I\n\n\nPlease note that there are no final answers provided by the system at the moment, as the",
        '```json\n{\n  "commands": [\n{\n"go to the baby rhinos": "You are standing on baby rhos",\ngo the berry:You are standing ber\ngo the ber:You are on berry\ngo to berr: You on ber\ngo ber: on ber: on\ngo: ber: onber you:\non yourber\no are ber\n\ngo ber on your\n ber\n on your ber\n to\ngo ber: on your ber\ngo bran\n\n onber on ber:You are on ber\nGo on berry: You on ber\non\nberYou on ber\non ber\ngo on:your:I am ber on ber\n\nber ongoing:You are onber\ngo onber: You on\n Go ber onyou are:on\nber on berY you\n on:are ber You\nber on beryou: on\n Gober you\n on',
        'One unique incorrectly translated action sequence and observation is "You go to the friends. You are standing on the friends."  \nThe next unique translated question sequence is "You go the water. You are standing the."',
        "a== The sequence of actions and observations when you go from the 'baby to the water' given the commands is: ['You are standing on water'.",
        "About In the sequence of actions: a: p mic as we are standing on the water o we are on the p: we standing the babyg o go the water o you standing the water o we on the p on now you are the bank water o On.\n\n   In the current:empty solo galax needing water a bank gogen. when you going on the p. We are on the p mostwaterbank. You on g standing on the r: go to banan the p. go the sleep your longestg p on the water o the water: The Romager leaning p going time as the: one the rinlong p the.\n the ri  oil going for the's banking on the go the blog o ic in garden on the bank on the we o gu you on the o water. The bank gow is going the the water o the cat. working the water now.\n\n\nAnswer:\nA shorter bearwomen bearing",
        'Rule 1:\nActions: go, stand\n\nObjects: baby, cow, water, pea, berry, giraffe, car, ria\n\n\nConstraint 1: Cannot be inside or outside of the given object.\n\nConstraint 2: Must follow the format: a).go to {object}\nConstraint 3: Must end with "o You are standing with/on {object}" or a go to {object} o You are standing/on {object}\nConstraint 4: Start with "a".\nConstraint 5: No not {follow }{ Rule.\nRule 1: max 2\n don\' Go Movement Conflict 1 Rule 1 max 1 Rule Max 1 max The sequence start with. Must avoid  In order  Go  unbut  Start Here Will  break Reason begin R conclusion\n\n Max  since end Break 2 end Even We Start  quote  EndAl The',
        "\"I started on the bird.\", you go the beet ball, You standing the. I. You on the water o: a go water: standing being on the water a: go cat: the yelap: We, going on a: Euz 'When, and all the water: Weel:o, you: aow, A cow:\nGo the: to watero: you. are standing the: Iog (2 Se): We stand on the : water: Our: standing the: Edge: A we on beh (Anything a You '. When looking on the I'Act for : searching the educational We a: I, go to the crowd: Awe:\nIn, dog.:the The.\n   You to animals a the: New: We\": Per: anymous.\nI am an before : commonly,\nWe: A.\n   You valuably: # Go thel if ox When: Suppose we the",
        # T-PB
        "In the current space: You see: baby sheep, water, beet seed, baby giraffe, water, pea seed, water, pea seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The pea appears from the transformation. a: You pick up the object. o: You are holding the pea. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You give the pea. o: The sheep appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The pea appears from the transformation. a: You pick up the object. o: You are holding the pea. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the pea and the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The beet appears from the transformation. a: You pick up the object. o: You are holding the pea and the beet. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give all the objects you hold. o: The giraffe appears from the transformation.",
        "In the current space: You see: baby cow, water, potato seed, baby giraffe, water, beet seed, water, potato seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the potato. a: You go to the baby cow. o: You are standing on the baby cow. a: You give the potato. o: The cow appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The beet appears from the transformation. a: You pick up the object. o: You are holding the beet. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the beet and the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the beet and the potato. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give all the objects you hold. o: The giraffe appears from the transformation.",
        "In the current space: You see: baby sheep, water, carrot seed, baby rhinoceros, water, potato seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The carrot appears from the transformation. a: You pick up the object. o: You are holding the carrot. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You give the carrot. o: The sheep appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the potato. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the potato and the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The carrot appears from the transformation. a: You pick up the object. o: You are holding the potato and the carrot. a: You go to the baby rhinoceros. o: You are standing on the baby rhinoceros. a: You give all the objects you hold. o: The rhinoceros appears from the transformation.",
        "In the current space: You see: baby cow, water, potato seed, baby giraffe, water, berry seed, water, berry seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the berry seed. o: You are standing on the berry seed. a: You give the water. o: The berry appears from the transformation. a: You pick up the object. o: You are holding the berry. a: You go to the baby cow. o: You are standing on the baby cow. a: You give the berry. o: The cow appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the berry seed. o: You are standing on the berry seed. a: You give the water. o: The berry appears from the transformation. a: You pick up the object. o: You are holding the berry. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the berry and the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the berry and the potato. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give all the objects you hold. o: The giraffe appears from the transformation.",
        "In the current space: You see: baby sheep, water, potato seed, baby rhinoceros, water, pea seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The carrot appears from the transformation. a: You pick up the object. o: You are holding the carrot. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You give the carrot. o: The sheep appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The pea appears from the transformation. a: You pick up the object. o: You are holding the pea. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the pea and the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the pea and the potato. a: You go to the baby rhinoceros. o: You are standing on the baby rhinoceros. a: You give all the objects you hold. o: The rhinoceros appears from the transformation.",
        "In the current space: You see: baby sheep, water, potato seed, baby elephant, water, beet seed, water, beet seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The beet appears from the transformation. a: You pick up the object. o: You are holding the beet. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You give the beet. o: The sheep appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The beet appears from the transformation. a: You pick up the object. o: You are holding the beet. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the beet and the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the beet and the potato. a: You go to the baby elephant. o: You are standing on the baby elephant. a: You give all the objects you hold. o: The elephant appears from the transformation.",
        "In the current space: You see: baby sheep, water, carrot seed, baby giraffe, water, potato seed, water, pea seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The pea appears from the transformation. a: You pick up the object. o: You are holding the pea. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You give the pea. o: The sheep appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the potato. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the potato and the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The carrot appears from the transformation. a: You pick up the object. o: You are holding the potato and the carrot. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give all the objects you hold. o: The giraffe appears from the transformation.",
        "In the current space: You see: baby pig, water, carrot seed, baby giraffe, water, beet seed, water, potato seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The potato appears from the transformation. a: You pick up the object. o: You are holding the potato. a: You go to the baby pig. o: You are standing on the baby pig. a: You give the potato. o: The pig appears from the transformation. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The beet appears from the transformation. a: You pick up the object. o: You are holding the beet. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the beet and the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The carrot appears from the transformation. a: You pick up the object. o: You are holding the beet and the carrot. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give all the objects you hold. o: The giraffe appears from the transformation.",
    ]
    algorithm_used: List[Optional[str]] = [
        None,
    ]
    all_rules_to_test.append(rules_to_test)
    algorithm_used = [None]
    for key in ["PB", "RB", "CP", "DB", "LL", "ALP", "T-PB"]:
        algorithm_used.extend([key] * 8)
    all_algorithm_used.append(algorithm_used)
    assert len(rules_to_test) == len(algorithm_used)
    # endregion

    # region Add finetuned LLM
    for model_path, algo_name in zip(
        args.finetuned_model_paths, args.finetuned_model_names
    ):
        configs.append(
            {
                "name": model_path,
                "use_unsloth": True,
                "model_params": {},
                "tokenizer_params": {},
                "is_quantized": True,
                "generation_kwargs": {
                    "cache_implementation": None,
                    "cache_config": None,
                },
                "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
            }
        )
        all_rules_to_test.append([None])
        all_algorithm_used.append([algo_name])
    # endregion
    # Main loop to evaluate the different world models
    all_scores = []
    all_index = []
    all_transitions_type = []
    print(f"{len(rules_to_test)} world models to evaluate.")
    for incr, (config, rules_to_test, algorithm_used) in enumerate(
        zip(configs, all_rules_to_test, all_algorithm_used)
    ):
        print("Evaluating model", config["name"])
        print(f"Number of rules to test: {len(rules_to_test)}")
        # Load the environment
        env: BaseWrapper = gymnasium.make(
            "worldllm_envs/PlaygroundText-v1",
            **{"seed": args.seed, "test_dataset_path": None, "max_steps": 30},
        )
        new_rule = env.unwrapped.generate_rule(
            "Grow any small_herbivorous then grow any big_herbivorous"
        )
        env.reset(options={"rule": new_rule})
        # Load the LLM
        world_model = load_transformers(config)
        stat_prompt_info = build_stat_prompt_info(
            world_model,
            env.unwrapped.get_message_info(),
            "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
            args.batch_size,
        )
        statistician = Statistician(
            world_model[0],
            world_model[1],
            stat_prompt_info,
        )
        # Add all the abstract trajectories
        statistician.prompt_info.discovered_transitions = (
            env.unwrapped.get_all_transition_to_prompt()
        )
        # Create the different agents
        perfect_agent_sh = PerfectAgent(
            env.action_space, curriculum_goals=["Grow any small_herbivorous"]
        )
        perfect_agent_shbh = PerfectAgent(
            env.action_space,
            curriculum_goals=[
                "Grow any small_herbivorous then grow any big_herbivorous"
            ],
        )
        random_agent = RandomAgent(env.action_space)
        nb_episodes = args.nb_episodes
        pbar = tqdm(range(3 * nb_episodes), desc="Generating trajectories")
        episode_counter = 0
        all_score_config = []
        for agent in [perfect_agent_sh, perfect_agent_shbh, random_agent]:
            for _ in range(nb_episodes):
                all_scores_per_episode = []
                all_index_per_episode = []
                all_transitions_type_per_episode = []
                obs, info = env.reset()
                reset_info = {"pipeline_progression": 0}
                info.update(reset_info)
                # Compute plan
                agent.reset(info)
                done = False
                while not done:
                    # Record inputs from keyboard
                    # Print possible actions
                    action, agent_done = agent(obs, **info)
                    all_transitions, transitions_type = (
                        env.unwrapped.get_all_possible_transitions()
                    )
                    # Get the score of the LLM for all transitions
                    lst_possible_trajectories = [
                        Trajectory(
                            lst_obs=[
                                info["trajectory_obs_text"][-1],
                                "",
                            ],  # We don't use the last observation
                            lst_act=[env.action_to_text(action)],
                            lst_diff=[transition],
                        )
                        for transition in all_transitions
                    ]
                    (_, logp), _ = compute_likelihood(
                        statistician,
                        rules_to_test,
                        lst_possible_trajectories,
                        return_all_logp=True,
                    )
                    all_scores_per_episode.append(logp.tolist())
                    all_transitions_type_per_episode.append(transitions_type)
                    obs, reward, terminated, truncated, info = env.step(action)
                    true_diff = info["trajectory_diff_text"][-1]
                    index_true = all_transitions.index(true_diff)
                    all_index_per_episode.append(index_true)
                    done = terminated or truncated or agent_done
                pbar.update(1)
                all_score_config.append(all_scores_per_episode)
                if incr == 0:
                    all_index.append(all_index_per_episode)
                    all_transitions_type.append(all_transitions_type_per_episode)
                else:
                    # Check if the same transitions
                    assert all_index_per_episode == all_index[episode_counter]
                    assert (
                        all_transitions_type_per_episode
                        == all_transitions_type[episode_counter]
                    )
                episode_counter += 1
        # Fusion score
        if incr == 0:
            all_scores = all_score_config
        else:
            all_scores = concatenate_third_axis(all_scores, all_score_config)

        # Empty gpu cache
        del statistician
        del stat_prompt_info
        del world_model
        gc.collect()
        torch.cuda.empty_cache()

    # Save the scores to json
    with open(args.output_name, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores": all_scores,
                "rules": flatten_list(all_rules_to_test),
                "algorithm_used": flatten_list(all_algorithm_used),
                "true_obs_index": all_index,
                "transition_type": all_transitions_type,
            },
            f,
        )

    print("Done.")
