import os

from dotenv import load_dotenv

load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

if not os.getenv("OPENROUTER_API_KEY"):
    print("OPENROUTER_API_KEY is not set")
    exit(1)

model = ChatOpenRouter(
    model="openai/gpt-5.2",
    temperature=1,
    max_tokens=1024,
    max_retries=2,

)
ollama_model=ChatOllama(
    model="gemma4",
    temperature=0.2,
    max_tokens=1024,
    max_retries=2,
)
anthropic_model=ChatAnthropic(
    model="claude-opus-4-7",
    max_tokens=1024,
    max_retries=2,
)
modiji_info="""Narendra Damodardas Modi was born on 17 September 1950 to a Gujarati family of Other Backward Class (OBC) background and Hindu faith[25][26] in Vadnagar, Mehsana district, Bombay State (present-day Gujarat). He was the third of six children born to Damodardas Mulchand Modi (c. 1915–1989) and Hiraben Modi (1923–2022).[27][h][28] According to Modi and his neighbours, he worked infrequently in his father's tea stall in the Vadnagar railway station.[29][30][31]

Modi completed his higher secondary education in Vadnagar in 1967; his teachers described him as an average student and a keen, gifted debater with an interest in theatre.[32] He preferred playing larger-than-life characters in theatrical productions, which has influenced his political image.[33][34]

When Modi was eight years old, he was introduced to the Rashtriya Swayamsevak Sangh (RSS) and began attending its local shakhas (training sessions). There, he met Lakshmanrao Inamdar, who inducted Modi as a balswayamsevak (junior cadet) in the RSS and became his political mentor.[35] While Modi was training with the RSS, he also met Vasant Gajendragadkar and Nathalal Jaghda, Bharatiya Jana Sangh leaders who in 1980 helped found the BJP's Gujarat unit.[36] As a teenager, he was enrolled in the National Cadet Corps.[37]

In a custom traditional to Narendra Modi's caste, his family arranged a betrothal to Jashodaben Chimanlal Modi, leading to their marriage when she was 17 and he was 18.[38][39] The marriage was never consummated, and Modi soon abandoned his wife,[40][41] and left home. The couple never divorced but the marriage was not in his public pronouncements for many decades.[39] In April 2014, shortly before the national election in which he gained power, Modi publicly affirmed he was married and that his spouse was Jashodaben.[42] A Modi biographer wrote that Modi kept the marriage a secret because he would not have been able to become a pracharak in the RSS, for which celibacy had once been a requirement.[43][44]

Modi spent the following two years travelling across northern and north-eastern India.[45] In mid 1968, Modi reached Belur Math but was turned away, after which he visited Calcutta, West Bengal and Assam, stopping in Siliguri and Guwahati. He then went to the Ramakrishna Ashram in Almora, where he was again rejected, before returning to Gujarat via Delhi and Rajasthan in 1968 to 1969. In either late 1969 or early 1970, he returned to Vadnagar for a brief visit before leaving again for Ahmedabad,[46][47] where he lived with his uncle and worked in his uncle's canteen at Gujarat State Road Transport Corporation.[48] Swami Vivekananda has had a large influence in Modi's life.[49]

In Ahmedabad, Modi renewed his acquaintance with Inamdar.[50][51][52] Modi's first-known political activity as an adult was in 1971 when he joined a Jana Sangh satyagraha in Delhi led by Atal Bihari Vajpayee to enlist to fight in the Bangladesh Liberation War.[53][54] The Indira Gandhi-led central government prohibited open support for the Mukti Bahini; according to Modi, he was briefly held in Tihar Jail.[55][56][57] After the Indo-Pakistani War of 1971, Modi left his uncle's employment and became a full-time pracharak (campaigner) for the RSS,[58] working under Inamdar.[59] Shortly before the war, Modi took part in a non-violent protest in New Delhi against the Indian government, for which he was arrested; because of this arrest, Inamdar decided to mentor Modi.[59] According to Modi, he was part of a satyagraha supporting the independence of Bangladesh.[56][i]

In 1978, Modi received a Bachelor of Arts (BA) degree in political science from the School of Open Learning[62] at the Delhi University.[43][63] In 1983, he received a Master of Arts (MA) degree in political science from Gujarat University, graduating with a first class[64][65] as an external distance learning student.[66] There is controversy surrounding the authenticity of his BA and MA degrees.[67][68][j]

Early political career
In June 1975, Prime Minister Indira Gandhi declared a state of emergency in India that lasted until 1977. During this period, known as "the Emergency", many of her political opponents were jailed and opposition groups were banned.[72][73] Modi was appointed general secretary of the "Gujarat Lok Sangharsh Samiti", an RSS committee coordinating opposition to the Emergency in Gujarat. Shortly afterwards, the RSS was banned.[74] Modi was forced to go underground in Gujarat and frequently travelled in disguise to avoid arrest, once dressing as a monk and once as a Sikh.[75] He became involved in the printing of pamphlets opposing the government, sending them to Delhi and organising demonstrations.[76][77] He was also involved with creating a network of safe houses for individuals who were wanted by the government, and in raising funds for political refugees and activists.[78] During this period, Modi wrote a Gujarati-language book titled Sangharsh Ma Gujarat (In the Struggles of Gujarat), which describes events during the Emergency.[79][80] While in this role, Modi met trade unionist and socialist activist George Fernandes and several other national political figures.[81]

Modi became an RSS sambhag pracharak (regional organiser) in 1978, overseeing activities in Surat and Vadodara, and in 1979, he went to work for the RSS in Delhi, where he researched and wrote the RSS's history of the Emergency. Shortly after, he returned to Gujarat and in 1985, the RSS assigned him to the BJP. In 1987, Modi helped organise the BJP's campaign in the Ahmedabad municipal election, which the party won comfortably; according to biographers, Modi's planning was responsible for the win.[82][83] After L. K. Advani became president of the BJP in 1986, the RSS decided to place its members in important positions within the party; Modi's work during the Ahmedabad election led to his selection for this role. Modi was elected organising secretary of the BJP's Gujarat unit later in 1987."""

summary_template="""
given the information {information} about a person, I want you to 
1. generate a 2 line summary of the person
2. 2 interesting facts about the person

the answer should only come from the information. dont guess or assume or use any other knowledge

"""

summary_prompt=PromptTemplate(template=summary_template, input_variables=["information"])

if __name__ == "__main__":
    # resp=model.invoke(summary_prompt.format(information=modiji_info)) option 1
    
    # option 2 Langchain expression language(LCEL)
    chain=summary_prompt | anthropic_model
    resp=chain.invoke(input=modiji_info)
    print(resp.content)
