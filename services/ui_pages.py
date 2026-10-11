from uuid import uuid4

import pandas as pd
import streamlit as st

import config_data as config
from services.chat_service import OfficeMateChatService
from services.document_service import DocumentService
from services.storage_service import JsonStorageService


def render_chat_page():
    storage = JsonStorageService()
    _render_api_key_notice()
    _init_chat_session(storage)
    _render_chat_sidebar(storage)

    st.title("OfficeMate：企业内部制度与流程智能助手")
    st.caption("面向企业内部制度、流程、通知与常见 IT 支持问题的轻量级知识助手。")

    if not storage.list_documents():
        st.info("当前知识库为空，请先前往“知识上传”页导入文档或示例知识库。")

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("qa_log_id"):
                _render_feedback_form(storage, message)

    prompt = st.chat_input("例如：报销差旅费需要提交哪些材料？")
    if prompt:
        question_category = st.session_state.get("selected_category", "全部")
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("正在检索制度与流程材料..."):
                    answer_result = OfficeMateChatService().answer_question(
                        question=prompt,
                        session_id=st.session_state["session_id"],
                        category=question_category,
                    )
                assistant_message = {
                    "role": "assistant",
                    "content": answer_result["answer"],
                    "qa_log_id": answer_result["qa_log_id"],
                    "question_type": answer_result["question_type"],
                    "question": prompt,
                    "domain": answer_result["domain"],
                    "search_category": answer_result["search_category"],
                }
                st.markdown(assistant_message["content"])
                st.caption(
                    f"已路由至 {answer_result['search_category']}，"
                    f"路由置信度 {answer_result['route_confidence']:.0%}"
                    f"{'，已触发全库回退' if answer_result['route_fallback'] else ''}"
                )
                _render_feedback_form(storage, assistant_message)
                st.session_state["chat_messages"].append(assistant_message)
            except Exception as exc:
                error_message = (
                    "### 最终回答\n当前回答失败，可能是模型 Key 未配置或知识库还没有完成初始化。\n\n"
                    "### 操作步骤/材料清单\n请检查 DASHSCOPE_API_KEY、知识库文档和网络连接。\n\n"
                    f"### 风险提示\n原始错误：{exc}"
                )
                st.error(error_message)


def render_upload_page():
    service = DocumentService()
    _render_api_key_notice()

    st.title("知识上传")
    st.caption("上传企业内部制度、流程、通知与 FAQ 文档，并补充分类、标题和版本信息。")

    with st.form("upload_form"):
        category = st.selectbox("文档分类", config.DOCUMENT_CATEGORIES)
        version = st.text_input("文档版本", value=config.DEFAULT_VERSION)
        custom_title = st.text_input("自定义标题（单文件上传时可选）")
        uploaded_files = st.file_uploader(
            "上传文档",
            type=config.SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
        )
        submit_upload = st.form_submit_button("导入知识库")

    if submit_upload:
        if not uploaded_files:
            st.warning("请先选择至少一个文件。")
        else:
            for uploaded_file in uploaded_files:
                effective_title = custom_title if len(uploaded_files) == 1 else ""
                result = service.ingest_uploaded_file(
                    uploaded_file=uploaded_file,
                    category=category,
                    version=version,
                    custom_title=effective_title,
                )
                _show_upload_result(result)

    st.divider()
    st.subheader("示例知识库")
    st.write("如果你暂时没有企业制度文档，可以先导入项目自带的示例文档进行演示。")
    if st.button("一键导入示例文档", key="seed_docs"):
        try:
            for result in service.seed_sample_documents():
                _show_upload_result(result)
        except Exception as exc:
            st.error(f"示例文档导入失败：{exc}")

    st.divider()
    st.subheader("最近导入文档")
    recent_docs = service.list_documents()[:10]
    if recent_docs:
        dataframe = pd.DataFrame(recent_docs)[
            ["title", "category", "version", "file_type", "chunk_count", "uploaded_at", "status"]
        ]
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
    else:
        st.info("还没有导入任何文档。")


def render_management_page():
    storage = JsonStorageService()
    document_service = DocumentService()
    stats = storage.get_stats()
    _render_api_key_notice()

    st.title("知识管理")
    st.caption("查看文档状态、问答记录和用户反馈，便于演示知识库闭环。")

    metric_columns = st.columns(4)
    metric_columns[0].metric("文档数量", stats["document_count"])
    metric_columns[1].metric("覆盖分类", stats["category_count"])
    metric_columns[2].metric("问答记录", stats["qa_count"])
    metric_columns[3].metric("反馈数量", stats["feedback_count"])
    st.caption(f"有效反馈中，好评率为 {stats['helpful_rate']:.0%}。")

    st.divider()
    st.subheader("文档列表")
    documents = storage.list_documents()
    if documents:
        document_frame = pd.DataFrame(documents)
        if "security_risk" not in document_frame:
            document_frame["security_risk"] = "unknown"
        document_frame = document_frame[
            [
                "title",
                "category",
                "version",
                "file_name",
                "chunk_count",
                "uploaded_at",
                "status",
                "security_risk",
                "source_label",
            ]
        ]
        st.dataframe(document_frame, use_container_width=True, hide_index=True)
    else:
        st.info("暂无文档记录。")

    st.subheader("删除已上传知识")
    if documents:
        document_options = {_build_document_option_label(document): document["id"] for document in documents}
        selected_label = st.selectbox(
            "选择需要删除的文档",
            options=list(document_options.keys()),
            key="delete_document_selector",
        )
        st.caption("删除后会同步移除原始文件和向量索引；历史问答与反馈记录会保留，便于继续演示使用痕迹。")
        confirm_delete = st.checkbox("我确认删除这份知识文档", key="confirm_delete_document")
        if st.button("删除选中文档", type="primary", key="delete_document_button"):
            if not confirm_delete:
                st.warning("请先勾选确认项，再执行删除。")
            else:
                result = document_service.delete_document(document_options[selected_label])
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
                elif result["status"] == "not_found":
                    st.warning(result["message"])
                else:
                    st.error(result["message"])
    else:
        st.info("当前没有可删除的文档。")

    st.divider()
    st.subheader("最近问答")
    qa_logs = storage.list_qa_logs(limit=20)
    if qa_logs:
        qa_frame = pd.DataFrame(qa_logs)
        qa_frame["source_count"] = qa_frame["source_docs"].apply(len)
        for column, default in (
            ("search_category", ""),
            ("top_relevance_score", 0.0),
            ("retrieval_strategy", "legacy"),
        ):
            if column not in qa_frame:
                qa_frame[column] = default
        qa_frame = qa_frame[
            [
                "created_at",
                "question_type",
                "search_category",
                "question",
                "source_count",
                "top_relevance_score",
                "retrieval_strategy",
            ]
        ]
        st.dataframe(qa_frame, use_container_width=True, hide_index=True)
    else:
        st.info("暂无问答记录。")

    st.divider()
    st.subheader("用户反馈")
    feedback_logs = storage.list_feedback()
    if feedback_logs:
        qa_lookup = {record["id"]: record for record in storage.list_qa_logs()}
        feedback_rows = []
        for feedback in feedback_logs:
            qa_log = qa_lookup.get(feedback["qa_log_id"], {})
            feedback_rows.append(
                {
                    "created_at": feedback.get("updated_at", feedback.get("created_at", "")),
                    "rating": _rating_to_label(feedback.get("rating", "")),
                    "comment": feedback.get("comment", ""),
                    "question": qa_log.get("question", ""),
                    "session_id": feedback.get("session_id", ""),
                }
            )
        st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无反馈记录。")


def _render_api_key_notice():
    if config.llm_provider == "ollama":
        return  # 本地模型不需要密钥
    if not config.get_api_key():
        st.warning(
            f"当前厂商为 {config.llm_provider}，未检测到对应的 API Key。"
            "页面可以打开，但模型调用会失败。"
        )


def _init_chat_session(storage):
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"{config.default_session_prefix}_{uuid4().hex[:8]}"

    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "全部"

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = _load_messages_from_logs(
            storage,
            st.session_state["session_id"],
        )
        if not st.session_state["chat_messages"]:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n你好，我是 OfficeMate。"
                        "你可以直接问我请假、报销、采购、IT 支持和通知总结等问题。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n我的回答以知识库中的制度与流程文档为准。"
                    ),
                }
            ]


def _render_chat_sidebar(storage):
    with st.sidebar:
        st.subheader("对话设置")
        st.selectbox(
            "知识范围",
            options=config.CATEGORY_FILTER_OPTIONS,
            key="selected_category",
        )
        st.caption(f"当前会话 ID：`{st.session_state['session_id']}`")

        if st.button("新建会话", key="new_session"):
            st.session_state["session_id"] = f"{config.default_session_prefix}_{uuid4().hex[:8]}"
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n已为你创建新会话，可以继续提问。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n如果需要准确回答，请先确认知识库中已有对应制度文档。"
                    ),
                }
            ]
            st.rerun()

        st.divider()
        st.subheader("推荐问题")
        st.markdown(
            "- 年假最晚需要提前几天申请？\n"
            "- 报销差旅费需要哪些材料？\n"
            "- 采购一台显示器应该怎么走流程？\n"
            "- VPN 连接失败怎么处理？"
        )

        st.divider()
        stats = storage.get_stats()
        st.caption(f"当前已导入 {stats['document_count']} 份文档，累计记录 {stats['qa_count']} 次问答。")


def _load_messages_from_logs(storage, session_id):
    messages = []
    for log in storage.list_session_logs(session_id):
        messages.append({"role": "user", "content": log["question"]})
        messages.append(
            {
                "role": "assistant",
                "content": log["answer"],
                "qa_log_id": log["id"],
                "question_type": log.get("question_type", ""),
                "question": log.get("question", ""),
            }
        )
    return messages


def _render_feedback_form(storage, message):
    qa_log_id = message.get("qa_log_id")
    if not qa_log_id:
        return

    existing_feedback = storage.get_feedback_by_qa_log_id(qa_log_id) or {}
    rating_options = ["未评价", "有帮助", "需改进"]
    default_rating = _rating_to_label(existing_feedback.get("rating", ""))
    default_index = rating_options.index(default_rating) if default_rating in rating_options else 0

    with st.expander("反馈", expanded=False):
        rating_label = st.radio(
            "这条回答是否有帮助？",
            rating_options,
            index=default_index,
            horizontal=True,
            key=f"rating_{qa_log_id}",
        )
        comment = st.text_input(
            "补充说明",
            value=existing_feedback.get("comment", ""),
            key=f"comment_{qa_log_id}",
        )
        if st.button("保存反馈", key=f"save_feedback_{qa_log_id}"):
            storage.upsert_feedback(
                qa_log_id=qa_log_id,
                rating=_label_to_rating(rating_label),
                comment=comment,
                session_id=st.session_state["session_id"],
            )
            st.success("反馈已保存。")


def _label_to_rating(label):
    return {
        "未评价": "unrated",
        "有帮助": "helpful",
        "需改进": "needs_improvement",
    }.get(label, "unrated")


def _rating_to_label(rating):
    return {
        "helpful": "有帮助",
        "needs_improvement": "需改进",
        "unrated": "未评价",
        "": "未评价",
    }.get(rating, "未评价")


def _show_upload_result(result):
    if result["status"] == "success":
        st.success(result["message"])
    elif result["status"] == "duplicate":
        st.info(result["message"])
    else:
        st.error(result["message"])


def _build_document_option_label(document):
    return (
        f"{document.get('title', '未命名文档')} | "
        f"{document.get('category', '未分类')} | "
        f"{document.get('version', '-')} | "
        f"{document.get('file_name', '-')}"
    )
