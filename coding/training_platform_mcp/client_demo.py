"""直接 MCP Client 示例；它尚不包含模型决策，因此不是完整 Agent。"""

import asyncio

from mcp import Client


async def main() -> None:
    async with Client("http://127.0.0.1:8000/mcp") as client:
        # Host 在连接后动态发现 Server 能力，不需要预先读取 Server 源码。
        tools_result = await client.list_tools()
        for tool in tools_result.tools:
            print(tool.name, tool.description)

        resources_result = await client.list_resources()
        for resource in resources_result.resources:
            print(resource.uri, resource.name)

        templates_result = await client.list_resource_templates()
        for template in templates_result.resource_templates:
            print(template.uri_template, template.name)

        prompts_result = await client.list_prompts()
        for prompt in prompts_result.prompts:
            print(prompt.name, prompt.description)

        # Prompt 由 Server 定义，Host 获取消息后才把它们加入 LLM 上下文。
        diagnosis_prompt = await client.get_prompt(
            "diagnose_training_job",
            {"job_id": "train-2048"},
        )
        print(diagnosis_prompt.messages)

        # Resource 由 Server 提供内容；Host 决定何时读取及是否加入模型上下文。
        error_manual = await client.read_resource("training://docs/error-codes")
        print(error_manual.contents)

        result = await client.call_tool("get_training_job", {"job_id": "train-2048"})
        if result.is_error:
            print("MCP 协议或工具调用失败：", result.content)
        else:
            # structured_content 适合 Runtime 稳定解析；content 主要供模型阅读。
            print(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
