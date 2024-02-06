import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
config();

const model = new ChatOpenAI();

export default model