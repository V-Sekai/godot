
.intel_syntax noprefix

    .text
    .align 2
.globl invokeNative
invokeNative:
    push rbp
    mov rbp, rsp

    mov r10, rcx    
    mov rax, rdx    
    mov rcx, r8     


    movsd xmm0, qword ptr [rax + 0]
    movsd xmm1, qword ptr [rax + 8]
    movsd xmm2, qword ptr [rax + 16]
    movsd xmm3, qword ptr [rax + 24]


    cmp rcx, 0
    jz cycle_end

    mov rdx, rsp
    and rdx, 15
    jz no_abort
    int 3
no_abort:
    mov rdx, rcx
    and rdx, 1
    shl rdx, 3
    sub rsp, rdx


    lea r9, qword ptr [rax + rcx * 8 + 64]
    sub r9, rsp 
cycle:
    push qword ptr [rsp + r9]
    loop cycle

cycle_end:
    mov rcx, [rax + 32]
    mov rdx, [rax + 40]
    mov r8,  [rax + 48]
    mov r9,  [rax + 56]

    sub rsp, 32 

    call r10
    leave
    ret





