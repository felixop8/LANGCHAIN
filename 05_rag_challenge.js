import { config } from "dotenv";
config();
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";
const model = new ChatOpenAI();
/* Challenge

    1. Create a retriever for the vectorStore. 
    
    2. Pass the correct values for context and personName into the prompt in the chain:
            - context: the pageContent of each documents obtained from the receiver, serialized 
              into a single string.
            - personName: the name of the person passed into the invocation of the chain. 

    3. Run the code and check the results in the console for result1 and result2. You should 
       receive accurate tips for how to make small talk with the two people.   
*/

const database = [
    {
      firstName: "John",
      lastName: "Smith",
      spouseName: "Mary",
      children: [
        {
          name: "Bobby",
          age: 12,
          interests: ["hockey", "drones", "Minecraft"],
        },
        {
          name: "Sally",
          age: 14,
          interests: ["lacrosse", "ukelele", "gymnastics"],
        },
      ],
      interests: ["golf", "tennis", "football"],
      job: "accountant",
    },
    {
      firstName: "Jane",
      lastName: "Doe",
      spouseName: "Sam",
      children: [
        {
          name: "Timmy",
          age: 5,
          interests: ["blocks", "bugs", "rocks"],
        },
        {
          name: "Tammy",
          age: 6,
          interests: ["dinosaurs", "tricycling", "ant farm"],
        },
      ],
      interests: ["cinema", "literature", "birdwatching"],
      job: "account executive",
    },
  ];

const stringifiedDatabase = database.map((record) => JSON.stringify(record));

  const vectorStore = await MemoryVectorStore.fromTexts(
    stringifiedDatabase,
    [{ id: 1 }, { id: 2 }],
    new OpenAIEmbeddings(),
)

const retriever = vectorStore.asRetriever()
const prompt =
  PromptTemplate.fromTemplate(`Tell me what can I talk about with {personName} to make it seem like I remember 
  personal details about their life. What is their job? What are their interests? What is their spouse's name? 
  What are their children's names, ages, and interests? Base the answers on the following context: 
{context}
`);

const serializeDocs = (docs) => docs.map((doc) => doc.pageContent).join("\n");

const chain = RunnableSequence.from([
    {
        context: retriever.pipe(serializeDocs),
        personName: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
]);


const result1 = await chain.invoke("John Smith and Jane Doe");
const result2 = await chain.invoke("Jane Doe");

console.log(result1);
console.log(result2);